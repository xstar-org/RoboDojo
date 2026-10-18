"""Wait for this launcher's model-ready marker after simulation initialization."""
import os
from pathlib import Path
import time


def wait_for_policy_ready():
    ready_file = os.environ.get("ROBODOJO_POLICY_READY_FILE")
    if not ready_file:
        return
    ready = Path(ready_file)
    timeout = float(os.environ.get("ROBODOJO_POLICY_READY_TIMEOUT", "600"))
    deadline = time.monotonic() + timeout
    print(f"[EvalEnv] scene initialized; waiting for policy READY: {ready}", flush=True)
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Policy model READY timeout after {timeout}s: {ready}")
        time.sleep(0.1)
