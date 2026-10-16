"""Launch independent copies of an evaluation client command for load testing.

This is a benchmarking utility only. Official evaluations use the standard
client startup unchanged; every worker here is a full, unmodified client that
replays the same task/layouts, so results are for throughput, not scoring.
"""
import argparse
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path


def stop_owned(children):
    """Terminate only process groups created by this invocation."""
    live = [p for p in children if p.poll() is None]
    for p in live:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for p in live:
        try:
            p.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()


def worker_env(base_env, run_id, worker_index, threads):
    env = dict(base_env, ROBODOJO_RUN_ID=run_id, EVAL_WORKER_INDEX=str(worker_index))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(threads)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be >= 1")
    if args.threads_per_worker < 1:
        parser.error("threads per worker must be positive")
    if os.name != "posix":
        parser.error("simulation process-group management requires Linux")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide the client command after --")

    args.output.mkdir(parents=True, exist_ok=False)
    run_base = f"{args.output.name}-{uuid.uuid4().hex[:12]}"
    plan = [dict(index=i, run_id=f"{run_base}-worker-{i}", command=list(command))
            for i in range(args.workers)]
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))

    children, logs = [], []
    try:
        for spec in plan:
            log = (args.output / f"worker-{spec['index']}.log").open("w")
            logs.append(log)
            child = subprocess.Popen(spec["command"],
                                     env=worker_env(os.environ, spec["run_id"], spec["index"],
                                                    args.threads_per_worker),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            (args.output / f"worker-{spec['index']}.pid").write_text(str(child.pid))
        while True:
            codes = [p.poll() for p in children]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"simulation client failed: {codes}")
            if all(code == 0 for code in codes):
                break
            time.sleep(0.2)
        (args.output / "DONE").write_text("All clients exited successfully; load-test results only.\n")
    except BaseException as error:
        (args.output / "FAILED").write_text(repr(error))
        raise
    finally:
        stop_owned(children)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
