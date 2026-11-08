# Native PT16 OpenWAM evaluation

This directory contains the runtime used to evaluate real OpenWAM checkpoints
with multiple isolated RoboDojo workers and resumable native case accounting.
The XPolicyLab submodule pins the tested `xstar-main` implementation.

The protocol uses three RGB cameras at 640×480, PathTracing with 16 samples per
pixel and total samples, OptiX denoising, 32 predicted/executed actions, and ten
denoising iterations. Checkpoint comparisons should use this same protocol.

## Optimizations

`runtime/measure.py` installs two verified optimizations:

* `counter_delivery.py` keeps the native two-delivery camera fence while bounding
  path-tracing accumulation to the current observation. It restores renderer
  settings and verifies that rendering does not advance physics.
* `gripper_snapshot.py` reuses native gripper readings within a control epoch,
  invalidates on every physics call, and checks cached values against native
  getters during the first 64 epochs. Callers retain independent array copies.

The runtime enables deferred USD writeback, cached tic-tac-toe geometry, and
skipping unused warmup observations. Scene setup still executes the native
physics sequence. `capture_eval_warmup_observations` defaults to true in the
environment, so the opt-in runtime controls this change.

## Runtime layout

The controller consumes an explicit run directory:

```text
run/
  serial_runner.py
  serial-multigpu-config.json
  task-plans.json
  parallel-cache-template.json
  RESUME_AUTHORIZATION.json
  <model>/
    fullsuite_controller.py
    balanced_model_plan.py
    multigpu_plan.py
    native_accounting.py
    process_identity.py
    manifest.json
    deploy.yaml
    foundation-deploy/
    group0/
      RoboDojo/                  # checkout with assets and SDK dependencies
      sim-python6-parallel      # environment-specific SDK launcher
      <runtime/*.py>
      tasks/<task>/
        partial.json
        native-exclusions.json
        result.json
```

Copy the controller modules and `runtime/*.py` into these locations. Launch
`serial_runner.py` inside the allocated job. The SDK launcher must activate the
Isaac Sim environment, select the physical render GPU and logical CUDA device,
and invoke `group0/instrument_entry.py` for the client entry point. Policy
inference uses the separate Python environment specified by the manifest.

`serial-multigpu-config.json` supplies `job`, `run_tag`, `model_order`, and
`held_tasks`. `task-plans.json` freezes each task's `seeds`, with a `seed` and
`layout_ids` list for each partition. `parallel-cache-template.json` supplies
`entries` as `[source_directory, destination_pattern]` pairs; patterns accept
`gpu` and `suffix` for private worker caches.

Each manifest supplies the allocated `cpus`, `policy_cpus`, `sim_cpu_pools`,
`policy_gpu`, `sim_gpus`, `workers_per_gpu` (three), `port`, `policy_python`,
`checkpoint_label`, `checkpoint_step`, `filesystem_preflight_passed`, the frozen
`task_plans_sha256`, and `resume_gate`/`resume_gate_sha256`. A model's
`foundation-deploy` directory provides the actual checkpoint, configuration,
normalization statistics and tokenizer expected by OpenWAM.

The explicit resume record contains `full_resume_allowed`, `resume_below100`,
`authorization_kind` (`explicit_user_full_evaluation_priority`), and
`performance_audit`. The referenced audit must establish
`mechanical_and_quality_passed`. This records the operator's choice to continue
evaluation below the 100 env-step/s tuning target; it does not report that the
target was achieved.

## Scheduling and preservation

`balanced_model_plan.py` partitions each seed into batches of at most 25 cases
and schedules the pending seed/layout keys across the manifest's simulation
GPUs. Existing valid and native-excluded keys are retained. Native outcomes keep
their source result path and index. Model execution is serial; the runner waits
for the previous model's processes to release their compute contexts.

The controller checks PID start time and cgroup before terminating owned
processes. An out-of-memory attempt reduces the batch size. A native GPU failure
stops new dispatch on that GPU and preserves collected rows. SDK-incompatible
tasks can remain explicitly pending in `held_tasks`; infrastructure errors are
recorded separately from policy outcomes.

The manifest, task plan, checkpoint identity, renderer settings and input hashes
should be retained alongside the native results. GPU roles come from the
manifest, allowing CUDA-only policy inference on a device whose rendering is
unavailable while healthy devices run simulation.

## Validation

Measured with a real HDF60000 checkpoint, 128 allocated logical CPUs, a 1200 GiB
memory allocation, one H100 for policy inference and two H100s for simulation:

| Workers per simulation GPU | Real envs per worker | 300 s env-step/s | Action-drained env-step/s | Native RGB comparisons |
|---|---:|---:|---:|---:|
| 3 | 25 | 67.58 | 71.04 | 450 |
| 2 | 32 | 61.33 | 63.26 | 384 |

Both rates include the first real model inference. These are bounded
`build_tower` performance probes; full-suite rates also include task startup and
turnover. The first configuration completed 24,800 actual env steps over 349.10
seconds; the second completed 22,528 over 356.13 seconds. A fourth allocated
GPU was unused in these probes.

Independent checks compared every camera against the native two-delivery
reference with frozen physics and an actual camera pose change. They verified
unchanged physics clocks, native physical-step and batch-IK counts, gripper
shadow equality, actual model batch shapes and 32-action chunks. A continuation
partition check retained 1,239 valid cases and 11 native exclusions and assigned
1,200 additional runnable keys without duplicates. This preserves the existing
HDF tic-tac-toe result of 28/50 successes and score 81.50.
