import os
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


def test_durable_crew_demo_runs_clean():
    proc = subprocess.run(
        [sys.executable, os.path.join("examples", "durable_crew_demo.py")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "demo passed: shared agent state" in proc.stdout