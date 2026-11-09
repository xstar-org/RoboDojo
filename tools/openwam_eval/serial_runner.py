"""Run the frozen model order with at most one model active."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "serial-multigpu-config.json").read_text())
child = None
stopping = False


def save(name, value):
    path = ROOT / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def stop(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
pending = {}
try:
    assert os.environ["SLURM_JOB_ID"] == str(CONFIG["job"])
    for model in CONFIG["model_order"]:
        assert not stopping
        directory = ROOT / model
        assert not subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        ).strip(), "Previous model still owns compute contexts"
        with (directory / "controller.log").open("a") as log:
            child = subprocess.Popen(
                ["/opt/conda/bin/python3", "-u", directory / "fullsuite_controller.py"],
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        proc = Path("/proc") / str(child.pid)
        save(
            "SERIAL_CURRENT.json",
            dict(
                time=time.time(),
                model=model,
                pid=child.pid,
                start=(proc / "stat").read_text().split(") ", 1)[1].split()[19],
                cgroup=(proc / "cgroup").read_text(),
            ),
        )
        code = child.wait()
        assert not stopping and code == 0, f"{model} controller exited: {code}"
        infrastructure = directory / "FULLSUITE_PENDING_INFRASTRUCTURE.json"
        if infrastructure.exists():
            pending[model] = json.loads(infrastructure.read_text())
            save(
                "SERIAL_PENDING_INFRASTRUCTURE.json",
                dict(time=time.time(), models=pending, all_requested_cases_complete=False),
            )
        else:
            assert (directory / "FULLSUITE_FINISHED.json").exists()
        deadline = time.monotonic() + 120
        while subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        ).strip():
            assert time.monotonic() < deadline, "Owned contexts did not exit"
            time.sleep(1)
    save(
        "SERIAL_AVAILABLE_FINISHED.json" if pending else "SERIAL_FINISHED.json",
        dict(time=time.time(), models=CONFIG["model_order"], pending=pending, all_requested_cases_complete=not pending),
    )
except BaseException:
    save("SERIAL_ERROR.json", dict(time=time.time(), error=traceback.format_exc()))
    raise
finally:
    save("SERIAL_STOPPED.json", dict(time=time.time()))
