import hashlib
import hmac
import json
import os
from pathlib import Path

from .actions import Action
from .crdts import from_dict


class JournalStore:
    """Durable, append-only journal with HMAC-signed lines and snapshots.

    Each journal line is `{"action": ..., "seq": N, "sig": hex}` when a
    ``journal_secret`` is configured; replay verifies every signature and
    refuses a tampered log. Snapshots are cumulative: every rotation folds the
    current root into the previous generation's root (lattice join) and emits
    a single cumulative summary, so the newest generation always contains the
    entire history up to its sequence number.
    """

    def __init__(self, data_dir, snapshot_threshold=1000, journal_secret=None):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.threshold = snapshot_threshold
        self.journal_secret = (journal_secret.encode("utf-8")
                               if isinstance(journal_secret, str) else journal_secret)
        self._log_path = self.dir / "journal.jsonl"
        self._snap_path = self.dir / "snapshot.json"
        self._seq = 0
        self._since_snapshot = 0
        self._log = None

    def _sign(self, payload):
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hmac.new(self.journal_secret, canonical, hashlib.sha256).hexdigest()

    def _verify(self, payload, sig):
        if self.journal_secret is None:
            return
        if not sig:
            raise ValueError("journal signature missing (tampered?)")
        if not hmac.compare_digest(self._sign(payload), str(sig)):
            raise ValueError("journal signature mismatch (journal tampered?)")

    def load(self):
        root = None
        seq = 0
        snap = self._read_snapshot()
        if snap is not None:
            if self.journal_secret is None and "sig" in snap:
                raise ValueError(
                    "snapshot is signed but no journal_secret was provided; "
                    "pass the signing key to verify it")
            self._verify({"seq": snap["seq"], "root": snap["root"]}, snap.get("sig"))
            seq = snap["seq"]
            root = from_dict(snap["root"])
        actions = []
        signed_entries = False
        if self._log_path.exists():
            lines = []
            with open(self._log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if line.strip():
                        lines.append(line)
            for idx, line in enumerate(lines, start=1):
                try:
                    entry = json.loads(line)
                except ValueError as exc:
                    # A crash mid-write can tear the final line; anything
                    # earlier is real corruption and must not be silently replayed.
                    if idx == len(lines):
                        break
                    raise ValueError(f"journal corrupt at line {idx}: {exc}") from exc
                if "sig" in entry:
                    signed_entries = True
                self._verify(entry["action"], entry.get("sig"))
                actions.append(Action.import_action(entry["action"]))
            if signed_entries and self.journal_secret is None:
                raise ValueError(
                    "journal lines are signed but no journal_secret was provided; "
                    "pass the signing key to verify them")
        self._seq = seq + len(actions)
        return root, actions

    def append(self, action):
        if self._log is None:
            self._log = open(self._log_path, "a", encoding="utf-8")
        entry = {"action": action.export(), "seq": self._seq + 1}
        if self.journal_secret is not None:
            entry["sig"] = self._sign(entry["action"])
        self._log.write(json.dumps(entry) + "\n")
        self._log.flush()
        os.fsync(self._log.fileno())
        self._seq += 1
        self._since_snapshot += 1
        return self._since_snapshot >= self.threshold

    def rotate(self, root_ormap):
        cumulative = self._cumulative_root(root_ormap)
        payload = {"seq": self._seq, "root": cumulative.to_dict()}
        if self.journal_secret is not None:
            payload["sig"] = self._sign({"seq": payload["seq"], "root": payload["root"]})
        self._write_snapshot(payload, self._seq)
        self._prune_generations()
        if self._log is not None:
            self._log.close()
            self._log = None
        self._log_path.unlink(missing_ok=True)
        self._since_snapshot = 0

    def _cumulative_root(self, root_ormap):
        previous = self._latest_generation_root()
        if previous is None:
            return root_ormap
        return previous.merge(root_ormap)

    def _latest_generation_root(self):
        generation = self._read_generation(self._newest_generation_seq())
        if generation is None:
            return None
        self._verify({"seq": generation["seq"], "root": generation["root"]},
                     generation.get("sig"))
        return from_dict(generation["root"])

    def _generation_files(self):
        return sorted(self.dir.glob("snapshot-*.json"),
                      key=lambda p: int(p.stem.partition("-")[2]))

    def _newest_generation_seq(self):
        files = self._generation_files()
        return int(files[-1].stem.partition("-")[2]) if files else None

    def _read_generation(self, seq):
        if seq is None:
            return None
        try:
            with open(self.dir / f"snapshot-{seq}.json", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _prune_generations(self):
        files = self._generation_files()
        for path in files[:-1]:
            path.unlink(missing_ok=True)

    def _write_snapshot(self, payload, seq):
        atomic = self.dir / f"snapshot-{seq}.json"
        tmp = atomic.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, atomic)
        tmp_latest = self._snap_path.with_suffix(".tmp")
        with open(tmp_latest, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_latest, self._snap_path)
        self._fsync_dir()

    def _fsync_dir(self):
        try:
            fd = os.open(self.dir, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _read_snapshot(self):
        if self._snap_path.exists():
            try:
                with open(self._snap_path, encoding="utf-8") as f:
                    return json.load(f)
            except ValueError:
                pass
        return self._read_generation(self._newest_generation_seq())

    def generations(self):
        return len(self._generation_files())

    def close(self):
        if self._log is not None:
            self._log.close()
            self._log = None