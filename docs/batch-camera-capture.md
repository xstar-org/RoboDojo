# Independent multi-environment camera capture

RoboDojo already exposes batched observations/actions and tracks termination per
environment. This change serializes the independent tiled camera render products
within a single physics state when `num_envs > 1`, so head and wrist products do
not interfere. Each camera is read after two render synchronization passes;
physics must not advance during capture. Camera resolution, scene configuration,
control, rewards and physics settings are unchanged. The env1 rendering path is
preserved. This is a correctness prerequisite, not a reduced-render-pass option.

Use the matching XPolicyLab VPP2 batch adapter with `eval_batch: true` and the
VPP2 backend batch API. Acceptance requires distinct valid head/left/right frames
for all requested environment IDs, including tails and terminated environments,
and task-score comparisons. Successful image decoding alone does not establish
equivalent camera inputs or task performance.
