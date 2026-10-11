"""Deterministic text fingerprints, similarity and local embeddings.

No numpy, no model, no network: the scoring used to group memories is a pure
function of the text, which keeps the whole cleanup pass reproducible (the same
two memories always get the same score) and lets exact and near-duplicate
detection work with no local LLM installed.

An optional vector is folded in when a caller (or a local model) supplies one.
Vectors produced here are *feature-hashed* bag-of-shingle embeddings: not
semantically deep, but cheap, stable across runs, and good enough to catch
paraphrases that share vocabulary. Any real embedding — sentence-transformers,
llama.cpp, an Ollama ``embed`` model — can be handed to ``MemoryStore.add`` and
is preferred when present.
"""

import hashlib
import math
import re
from difflib import SequenceMatcher
import unicodedata
from array import array
from dataclasses import dataclass, field

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s'-]", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+|\n+")
_NUMBER_RE = re.compile(r"-?\d[\d,_]*(?:\.\d+)?|\d+/\d+")
_NEGATIONS = frozenset({
    "not", "no", "never", "none", "n't", "cannot", "without", "false", "avoid",
    "prohibited", "forbidden", "disallowed",
})
_AFFIRMATIONS = frozenset({"always", "must", "yes", "true", "required", "enabled", "only"})

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in",
    "is", "it", "its", "just", "may", "my", "not", "of", "on", "or", "please", "should",
    "so", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "was", "we", "were", "what", "when", "which", "who", "will", "with",
    "you", "your",
})


def normalize_text(text):
    """Canonical form used for fingerprints and comparison.

    Unicode-normalized, casefolded, punctuation-insensitive and whitespace
    collapsed — so ``"Deploy  the API, first!"`` and ``"deploy the api first"``
    collide on purpose.
    """
    if text is None:
        return ""
    out = unicodedata.normalize("NFKC", str(text))
    out = out.replace("\u2019", "'").replace("\u2018", "'")
    out = out.replace("\u201c", '"').replace("\u201d", '"')
    out = _PUNCT_RE.sub(" ", out)
    out = _WS_RE.sub(" ", out).strip()
    return out.casefold()


def fingerprint(*parts):
    """Stable content hash used for exact-duplicate detection."""
    joined = "\x1f".join(str(part) for part in parts if part is not None)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def content_fingerprint(body, kind=None, topic=None):
    """Fingerprint of a memory's content: body plus its grouping keys.

    Two memories with the same body but a different ``kind`` are not exact
    duplicates — the kind is part of what the record means, and folding them
    together would quietly drop it.
    """
    return fingerprint(normalize_text(body), kind, topic)


def tokens(text):
    return [t for t in normalize_text(text).split(" ") if t]


def content_words(text):
    return [t for t in tokens(text) if t not in STOPWORDS]


def word_shingles(text, size=3):
    """Word-level n-gram set (unigrams, bigrams and up to ``size``-grams)."""
    words = tokens(text)
    if not words:
        return frozenset()
    grams = set(words)
    for n in range(2, max(2, int(size)) + 1):
        if len(words) < n:
            continue
        for i in range(len(words) - n + 1):
            grams.add(" ".join(words[i:i + n]))
    return frozenset(grams)


def char_shingles(text, size=4):
    """Character n-grams — sensitive to typos and reworded tails."""
    clean = normalize_text(text).replace(" ", "_")
    if not clean:
        return frozenset()
    if len(clean) < size:
        return frozenset({clean})
    return frozenset(clean[i:i + size] for i in range(len(clean) - size + 1))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    overlap = sum(1 for item in a if item in b)
    return overlap / (len(a) + len(b) - overlap)


def containment(a, b):
    """Asymmetric overlap: how much of ``a`` is covered by ``b``.

    A one-line memory restated inside a longer one scores low on Jaccard but
    high on containment — exactly the case a merge should catch.
    """
    if not a:
        return 1.0 if not b else 0.0
    if not b:
        return 0.0
    return sum(1 for item in a if item in b) / len(a)


@dataclass(frozen=True)
class Signature:
    """Everything the scorer needs from one memory, computed once."""

    norm: str = ""
    words: frozenset = frozenset()
    grams: frozenset = frozenset()
    chars: frozenset = frozenset()
    vector: tuple = field(default=None, repr=False)

    def blocking_keys(self):
        """Inverted-index keys this signature should be bucketed under."""
        return self.grams | self.words


def signature(body, vector=None):
    norm = normalize_text(body)
    return Signature(
        norm=norm,
        words=frozenset(content_words(norm)),
        grams=word_shingles(norm),
        chars=char_shingles(norm),
        vector=None if vector is None else tuple(float(x) for x in vector),
    )


def text_similarity(sa, sb):
    """Symmetric similarity in [0, 1] between two :class:`Signature` objects."""
    if sa.norm == sb.norm:
        return 1.0
    if not sa.norm or not sb.norm:
        return 0.0
    base = max(jaccard(sa.chars, sb.chars), jaccard(sa.grams, sb.grams))
    if base <= 0.0:
        return 0.0
    cover = max(containment(sa.words, sb.words), containment(sb.words, sa.words))
    return min(1.0, base + 0.3 * cover * (1.0 - base))


def cosine(a, b):
    if a is None or b is None:
        return None
    if len(a) != len(b):
        return None
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def blend(text_score, vector_score, text_weight=0.75, vector_weight=0.25):
    """Fold the vector score into the text score, renormalizing when absent."""
    if vector_score is None:
        return text_score
    total = float(text_weight) + float(vector_weight)
    if total <= 0:
        return text_score
    return (float(text_weight) * text_score + float(vector_weight) * vector_score) / total


def similarity(sa, sb, text_weight=0.75, vector_weight=0.25):
    """Blended text/vector score for a pair of signatures."""
    return blend(text_similarity(sa, sb), cosine(sa.vector, sb.vector), text_weight, vector_weight)


def split_sentences(text):
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(str(text).strip()) if p and p.strip()]
    return parts or ([str(text).strip()] if str(text).strip() else [])


def numbers_in(text):
    return [n.replace(",", "").replace("_", "") for n in _NUMBER_RE.findall(str(text))]


def polarity(text):
    """Rough negation signal: -1 negative, 1 positive, 0 undecided."""
    words = set(tokens(text))
    negative = bool(words & _NEGATIONS)
    positive = bool(words & _AFFIRMATIONS)
    if negative and not positive:
        return -1
    if positive and not negative:
        return 1
    return 0


def substitutions(sa, sb):
    """Same-shape pairs where one content word was swapped, e.g. Alice/Bob.

    Returns the list of ``(a_word, b_word)`` replacements, or ``None`` when the
    texts differ by an insertion or deletion — a restatement, which is exactly
    what a merge is for.
    """
    a, b = sa.norm.split(), sb.norm.split()
    if not a or not b:
        return None
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and (i2 - i1) == (j2 - j1) == 1:
            out.append((a[i1], b[j1]))
            continue
        return None
    return out or None


def conflicts(sa, sb):
    """True when two similar memories cannot be folded into one statement.

    Different numbers, opposite polarity, or a swapped content word all mean a
    merge would destroy information, so the pair is reviewed instead. Purely
    length-changing differences are *not* conflicts: that is a restatement.
    """
    if sa.norm == sb.norm:
        return False
    an, bn = set(numbers_in(sa.norm)), set(numbers_in(sb.norm))
    # Disjoint number sets ("30 min" vs "45 min") contradict each other. A superset
    # does not: the longer record adds a measurement the shorter one never made, and
    # the fold keeps both sentences, so nothing is lost.
    if an != bn and (an or bn) and not (an <= bn or bn <= an):
        return True
    pa, pb = polarity(sa.norm), polarity(sb.norm)
    if pa and pb and pa != pb:
        return True
    for left, right in substitutions(sa, sb) or ():
        if left not in STOPWORDS and right not in STOPWORDS:
            return True
    return False


# --- local vector model ------------------------------------------------------


def hash_embed(text, dim=256, salt=""):
    """Deterministic feature-hashed unit vector over word shingles.

    Signed hashing (the "hashing trick"): every shingle adds +1 or -1 (decided
    by a second byte of the digest) to one bucket, then the vector is L2-normed
    so cosine is a plain dot product.
    """
    dim = int(dim)
    if dim < 8:
        raise ValueError("embedding dim must be at least 8")
    vec = [0.0] * dim
    grams = word_shingles(text) or {normalize_text(text)}
    for gram in sorted(grams):
        digest = hashlib.sha256(f"{salt}\x1f{gram}".encode("utf-8")).digest()
        vec[int.from_bytes(digest[:8], "big") % dim] += 1.0 if digest[8] % 2 == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return array("f", [0.0] * dim)
    return array("f", [round(v / norm, 6) for v in vec])


def pack_vector(vector):
    if vector is None:
        return None
    return array("f", (float(v) for v in vector)).tobytes()


def unpack_vector(blob, dim=None):
    if not blob:
        return None
    vec = array("f")
    vec.frombytes(bytes(blob))
    if dim is not None and len(vec) != int(dim):
        return None
    return vec


def fold_unique_sentences(bodies, max_chars=700, prefix="[merged]"):
    """Model-free merge: union of sentences, most recent source first.

    Already-present sentences are dropped so the merged body never repeats a
    statement. Nothing is lost by truncation at the record level — every source
    memory is retained (only its state changes) and its full body stays in the
    audit trail.
    """
    seen = set()
    kept = []
    used = 0
    for body in bodies:
        for sentence in split_sentences(body):
            key = fingerprint(normalize_text(sentence))
            if key in seen:
                continue
            seen.add(key)
            cost = len(sentence) + (1 if kept else 0)
            if used + cost > max_chars:
                continue
            if not re.search(r"[.!?。！？]$", sentence):
                sentence = sentence + "."
            kept.append(sentence)
            used += cost
    merged = " ".join(kept) if kept else " ".join(str(b).strip() for b in bodies)[:max_chars]
    if prefix:
        merged = f"{prefix} {merged}"
    return merged
