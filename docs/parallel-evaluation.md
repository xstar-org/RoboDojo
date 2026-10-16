# Parallel Load Testing (utility only)

`scripts/eval_parallel.py` launches N independent, unmodified evaluation clients
against one already-running policy server. It exists **only for throughput/load
testing**; official evaluations use the standard startup path unchanged.

## What it does

- Spawns the given client command N times verbatim — no flags are injected and
  `src/eval_client/main.py` is not modified in any way.
- Per worker: unique `ROBODOJO_RUN_ID` (`<output>-<uuid>-worker-i`),
  `EVAL_WORKER_INDEX`, thread caps, `worker-i.log/.pid`, own process group.
- Failure of any worker stops and reaps **only** the process groups this
  invocation created; the shared policy server is never touched.

## Layouts are intentionally repeated

Every worker is a full client for the same task and seed, so all workers run
the same layout list. That is the desired behaviour for a load test: it
measures saturation throughput without consuming unique evaluation layouts
(a task/seed has at most 75 anyway). Results from this path are throughput
numbers, never success-rate/score evidence.

## Usage

```bash
bash scripts/robodojo.sh client ... &   # or any standard client command
python3 scripts/eval_parallel.py --workers 4 --output runs/load4 \
  -- bash scripts/internal/run_policy_eval.sh <policy-dir> <args...>
```

Reference measurement (Job307, 4×H100, VPP2 offload companion, server
micro-batch 32): 4 workers × 32 envs (128 envs total) reached 7.67 total
steps/s vs 5.22 for one worker × 32 envs on the same stack.
