"""Reuse unchanged native gripper readings only within a control/physics epoch.

No cache survives a physics call or control-step exit. The native getter still
produces the values; this never changes targets, interpolation, or physics.
"""

import numpy as np


def install(env, *, enabled, shadow_epochs=64):
    stats = dict(
        enabled=enabled,
        requests=0,
        batch_reads=0,
        hits=0,
        physics_invalidations=0,
        shadow_calls=0,
        shadow_epochs=0,
        max_abs_error=0.0,
    )
    env._gripper_snapshot_stats = stats
    if not enabled:
        return stats
    manager = env.robot_manager
    native_get = manager.get_end_effector_real_val
    native_step = env.step
    native_physics = env.sim_step
    cache = {}
    state = dict(active=False, epoch=0)
    shadowed = set()
    epochs_checked = set()

    def get(robot, env_idx_list=None):
        if not state["active"]:
            return native_get(robot, env_idx_list=env_idx_list)
        stats["requests"] += 1
        key = id(robot)
        hit = key in cache
        if not hit:
            cache[key] = native_get(robot, env_idx_list=None)
            stats["batch_reads"] += 1
        else:
            stats["hits"] += 1
        values = cache[key]
        # Keep native dictionary keys/order, missing selections, and copy
        # ownership. Callers may mutate the returned arrays.
        result = {
            i: (None if v is None else v.copy()) if env_idx_list is None or i in env_idx_list else None
            for i, v in values.items()
        }
        check = (state["epoch"], key)
        if hit and state["epoch"] < shadow_epochs and check not in shadowed:
            reference = native_get(robot, env_idx_list=env_idx_list)
            assert reference.keys() == result.keys()
            for i, actual in result.items():
                expected = reference[i]
                if actual is None or expected is None:
                    assert actual is None and expected is None
                else:
                    assert np.array_equal(actual, expected), "Gripper snapshot differs from native state"
                    if actual.size:
                        error = float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
                        stats["max_abs_error"] = max(stats["max_abs_error"], error)
            shadowed.add(check)
            epochs_checked.add(state["epoch"])
            stats["shadow_calls"] += 1
            stats["shadow_epochs"] = len(epochs_checked)
        return result

    def step(*args, **kwargs):
        assert not state["active"], "Nested control steps are unsupported"
        cache.clear()
        state["active"] = True
        try:
            return native_step(*args, **kwargs)
        finally:
            state["active"] = False
            cache.clear()

    def physics(*args, **kwargs):
        cache.clear()
        active = state["active"]
        try:
            return native_physics(*args, **kwargs)
        finally:
            cache.clear()
            if active:
                state["epoch"] += 1
                stats["physics_invalidations"] += 1

    manager.get_end_effector_real_val = get
    env.step = step
    env.sim_step = physics
    return stats
