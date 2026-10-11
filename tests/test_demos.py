import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_agent_crew_demo_runs_clean():
    proc = subprocess.run(
        [sys.executable, os.path.join("examples", "agent_crew_demo.py")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "demo passed: signed agent crew" in proc.stdout


def test_self_pruning_memory_demo_runs_clean():
    proc = subprocess.run(
        [sys.executable, os.path.join("examples", "self_pruning_memory_demo.py")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "demo passed: self-pruning memory" in proc.stdout
    # the story it tells has to include the parts that make it safe: nothing is
    # deleted, a contradiction is refused, an unknown knob is refused, the mirror
    # is fingerprinted
    for expected in ("nothing deleted", "unknown policy keys", "state_hash=",
                     "conflict=True"):
        assert expected in proc.stdout, expected


def test_durable_crew_demo_runs_clean():
    proc = subprocess.run(
        [sys.executable, os.path.join("examples", "durable_crew_demo.py")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "demo passed: shared agent state" in proc.stdout


def _memory_snippet():
    """The python block under '## Self-pruning memory layer' in README.md."""
    with open(os.path.join(REPO_ROOT, "README.md"), encoding="utf-8") as handle:
        text = handle.read()
    section = text.split("## Self-pruning memory layer", 1)[1]
    block = section.split("```python", 1)[1].split("```", 1)[0]
    return block.strip("\n") + "\n"


def test_readme_memory_snippet_actually_runs(tmp_path):
    """The README example is executed, not eyeballed, and prints what it claims.

    A copy-pasteable block is a promise about the public API's shape, and the
    memory layer's surface moved a lot while its docs were being written. The
    snippet is rewritten only so it runs against a scratch database.
    """
    code = _memory_snippet()
    assert "CleanupService" in code and "apply_run" in code
    # the snippet states its own expected output, so a stale doc fails loudly
    claimed = [line for line in code.splitlines()
               if line.lstrip().startswith("# ") and re.search(r"\d", line)]
    assert claimed, "the README snippet should state what it prints"

    body, _, tail = code.partition("svc.close()")
    script = tmp_path / "snippet.py"
    script.write_text(body.replace('CleanupService("memories.db"',
                                  'CleanupService(r"' + str(tmp_path / "mem.db") + '"')
                      + "\nsvc.close()\n" + tail, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=120, cwd=REPO_ROOT,
                          env={**os.environ, "PYTHONPATH": REPO_ROOT})
    assert proc.returncode == 0, proc.stderr[-2000:]
    printed = [line.strip() for line in proc.stdout.strip().splitlines()]
    assert printed[:2] == ["1 1 1", "contradiction"], printed
