"""Independent process-level oracle for the Macaulay2 regression script."""

from pathlib import Path
import os
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
test_script = ROOT / "tests" / "constructive_lambda_test.m2"
m2 = os.environ.get("M2") or shutil.which("M2")

if not m2:
    raise SystemExit("M2 is required; set M2=/path/to/M2")

completed = subprocess.run(
    [m2, "--no-readline", str(test_script)],
    cwd=ROOT,
    text=True,
    capture_output=True,
    check=False,
)

if completed.returncode != 0:
    sys.stderr.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    raise SystemExit(completed.returncode)

required = [
    "PASS: Experiment 1 bounded count",
    "PASS: Experiment 2 bounded count",
    "PASS: Experiment 3 bounded count",
    "ALL CONSTRUCTIVE LAMBDA TESTS PASSED",
]
missing = [marker for marker in required if marker not in completed.stdout]
if missing:
    sys.stderr.write(completed.stdout)
    raise SystemExit("missing Macaulay2 markers: " + ", ".join(missing))

print("PASS: independent Macaulay2 process oracle")
