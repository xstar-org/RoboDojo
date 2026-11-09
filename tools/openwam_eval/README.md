# OpenWAM evaluation with Isaac Sim 6

This opt-in runtime runs real OpenWAM checkpoints with one policy GPU and up to
three simulation GPUs. It preserves native scoring, physics and unstable-case
exclusions. The validated SDK baseline is **Isaac Sim 6.0.0.1**; this is not a
claim of validation on 6.1.

The evaluation protocol is three 640×480 RGB cameras, PathTracing at 16 samples
per pixel and total samples, OptiX denoising, two native camera deliveries,
32 predicted/executed actions and ten denoising iterations. The XPolicyLab
submodule pins the tested implementation.

## Entry points

`runtime/instrument_entry.py` runs the standard evaluation client with its
explicit `--env_factory_module client_runtime` hook. It does not rewrite or
execute a modified copy of the client's source. `client_runtime.py` selects the
worker's layouts on its own seed manager, records actual counters and input
identities, installs the optimizations and checks image/physics invariants.

The two simulation optimizations are intentionally isolated:

- `counter_delivery.py` bounds accumulation to the current observation, retains
  two native camera deliveries, restores renderer settings even on error and
  verifies that capture does not advance physics.
- `gripper_snapshot.py` reuses native gripper readings within a control epoch,
  invalidates on every physics call and control-step exit, and compares cache
  hits with native getters during the first 64 epochs. Returned arrays are copies.

Deferred USD writeback, cached tic-tac-toe geometry and skipping discarded
warmup observations are enabled only by this runtime. Native scene setup and
control execution are retained. First-frame RGB checks, setup physics counts,
model batch shapes and 32-action chunks are recorded alongside results.

`runtime/server_entry.py` isolates each worker's OpenWAM history while sharing
the model. `profile_policy.py` records real model timings and tensor shapes;
there is no fixed-worker startup barrier. `policy_compat.py` contains the pinned
OpenWAM dependency's prompt-cache eviction fix. It cannot simply be deleted:
the dependency's getter changes LRU order during eviction. Remove this adapter
when that fix is available in the pinned dependency.

## Run directory

Stage the files in the allocated job as follows:

```text
run/
  serial_runner.py
  serial-multigpu-config.json
  task-plans.json
  parallel-cache-template.json
  <model>/
    fullsuite_controller.py
    balanced_model_plan.py
    multigpu_plan.py
    native_accounting.py
    manifest.json
    deploy.yaml
    foundation-deploy/
    group0/
      RoboDojo/                  # checkout, assets and SDK dependencies
      sim-python6-parallel      # environment-specific SDK launcher
      <runtime/*.py>
      tasks/<task>/             # native results and resumable case ledger
```

The SDK launcher selects the physical rendering GPU and logical CUDA device,
activates Isaac Sim and invokes `instrument_entry.py`. The manifest supplies
`job` (string), `cpus`, `policy_cpus`, `sim_cpu_pools`, `policy_gpu`, `sim_gpus`,
`workers_per_gpu`, `max_envs_per_worker` (1–25), `port`, `policy_python`,
`checkpoint_label`, `checkpoint_step`, `filesystem_preflight_passed` and
`task_plans_sha256`. `foundation-deploy` contains the checkpoint, configuration,
normalization statistics and tokenizer. CPU pools must exclude policy CPUs.
GPU roles are explicit; no GPU number or physical topology is assumed.

`task-plans.json` maps task names to `seeds`, each with `seed` and `layout_ids`.
Task and case totals are derived from this frozen plan. The configuration supplies
`job`, `run_tag`, `model_order` and optional `held_tasks`, `priority_tasks`,
`task_batch_sizes` and `complete_first_task`. For example, a batch limit of six
partitions a 50-case seed into five six-case and four five-case workers;
`complete_first_task: play_tic_tac_toe` defers other tasks until those cases finish.
`parallel-cache-template.json` supplies `entries` as source/destination pairs;
destination patterns accept `gpu` and `suffix` for private worker caches.

Run `serial_runner.py` inside the allocation. No experiment-specific approval,
performance-audit or 100 env-step/s gate file is required.

## Resume and failures

Only pending seed/layout keys are scheduled. Completed rows and native exclusions
retain result paths and indices across restarts. An incomplete reply stays pending;
it does not become a policy failure. Known SDK-incompatible tasks can be listed in
`held_tasks`. An out-of-memory attempt reduces batch size; a native GPU failure
stops dispatch on that GPU. Termination checks PID start time and cgroup before
signalling owned processes. Checkpoints run serially after previous processes exit.
Resume uses saved native case ledgers; live-process adoption is not supported.

## Validation

CPU regression checks (NumPy required):

```bash
python -m unittest discover -s tools/openwam_eval/tests -v
```

They cover restart-safe client hook loading, camera-fence restoration on errors,
exact case partition/resume, partial metric writes and resets, native
exclusions, per-environment layout filtering, gripper cache invalidation/copy
ownership, prompt-cache LRU eviction and policy-history isolation between workers.

The simulation mechanisms were previously checked against native frozen-physics
RGB references and actual camera pose changes on SDK 6.0.0.1. A real HDF60000
`build_tower` probe with two simulation H100s, three workers × 25 environments
per GPU and one policy H100 measured 67.58 env-step/s over 300 seconds and
71.04 including action drain and initial inference; 450 RGB comparisons passed.
These are historical single-task measurements, not full-suite throughput or a
new native validation of this refactor. Existing allocated evaluations continue
using their staged code; this PR update does not modify those running copies.
