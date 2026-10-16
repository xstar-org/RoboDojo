"""Tests for the load-test launcher. Run on Linux for process-group coverage."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "eval_parallel.py"


class EvalParallelTests(unittest.TestCase):
    def run_launcher(self, *args, timeout=60):
        return subprocess.run([sys.executable, str(LAUNCHER), *args],
                              capture_output=True, text=True, timeout=timeout)

    def test_command_is_passed_verbatim(self):
        """The worker command must not be augmented with layout/env flags."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            rc = self.run_launcher("--workers", "2", "--output", str(out), "--",
                                   "python3", "-c", "import sys; sys.exit(0)")
            self.assertEqual(rc.returncode, 0, rc.stderr)
            plan = json.loads((out / "plan.json").read_text())
            self.assertEqual(len(plan), 2)
            for spec in plan:
                self.assertEqual(spec["command"], ["python3", "-c", "import sys; sys.exit(0)"])

    def test_worker_envs_and_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            code = ("import os,sys;"
                    "print(os.environ['ROBODOJO_RUN_ID'], os.environ.get('EVAL_WORKER_INDEX'), os.environ['OMP_NUM_THREADS']);"
                    "sys.exit(0)")
            rc = self.run_launcher("--workers", "3", "--threads-per-worker", "2",
                                   "--output", str(out), "--",
                                   sys.executable, "-c", code)
            self.assertEqual(rc.returncode, 0, rc.stderr)
            run_ids = set()
            for i in range(3):
                text = (out / f"worker-{i}.log").read_text().split()
                self.assertEqual(text[1], str(i))
                self.assertEqual(text[2], "2")
                run_ids.add(text[0])
            self.assertEqual(len(run_ids), 3)
            self.assertTrue((out / "DONE").exists())

    @unittest.skipUnless(os.name == "posix", "process groups are Linux-only")
    def test_failure_cleans_only_owned_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            keeper = subprocess.Popen(["sleep", "300"], start_new_session=True)
            try:
                code = "import os,time,signal,sys; time.sleep(30)"
                rc = self.run_launcher("--workers", "2", "--output", str(out), "--",
                                       sys.executable, "-c", code, timeout=30)
                self.assertNotEqual(rc.returncode, 0)
                self.assertTrue((out / "FAILED").exists())
                pids = [int((out / f"worker-{i}.pid").read_text()) for i in range(2)]
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and any(
                        Path(f"/proc/{p}").exists() for p in pids):
                    time.sleep(0.2)
                for p in pids:
                    self.assertFalse(Path(f"/proc/{p}").exists(), f"worker {p} not cleaned")
            finally:
                os.killpg(keeper.pid, 15)
                keeper.wait()


import time  # noqa: E402  (used in the posix test above)

if __name__ == "__main__":
    unittest.main()
