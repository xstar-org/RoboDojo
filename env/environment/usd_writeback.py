"""Publish CPU physics transforms before consumers, retaining every physics step."""
from contextlib import contextmanager


USD_WRITEBACK = "/physics/updateToUsd"


@contextmanager
def deferred_usd_writeback(env, env_ids):
    """Optionally defer intermediate USD output during the control queue drain.

    Control/safety reads use native physics tensors. Tasks with custom per-step
    behavior or interval events retain their original writeback behavior.
    """
    if not env.sim_config.get("defer_usd_writeback", False):
        yield
        return

    from carb.settings import get_settings
    from env.environment.task_env import TaskEnv

    modes = getattr(getattr(env.sim, "event_manager", None), "available_modes", [])
    if (str(env.sim.sim.device) != "cpu" or env.use_fabric
            or getattr(type(env).__mro__[1], "step", None) is not TaskEnv.step
            or "interval" in modes):
        yield
        return

    settings = get_settings()
    saved = settings.get(USD_WRITEBACK)
    if not saved:
        yield
        return
    if getattr(env, "_usd_writeback_active", False):
        raise RuntimeError("Nested deferred physics controls are unsupported")

    stats = getattr(env, "usd_writeback_stats", None)
    if stats is None:
        stats = env.usd_writeback_stats = dict(
            action_batches=0, simulation_groups=0,
            suppressed_groups=0, published_groups=0, exceptions=0,
        )
    stats["action_batches"] += 1
    ids = list(env_ids)
    queues = env.robot_manager.control_manager.control_queue
    original = env.sim_step
    had_override = "sim_step" in vars(env)
    pending = False

    def step(*args, **kwargs):
        nonlocal pending
        # EvalEnv.step pops the controls before invoking sim_step. Do not call
        # get_empty(): it also clears inactive queues as a side effect.
        final = not ids or any(queues[i].is_empty() for i in ids)
        desired = saved if final else False
        if settings.get(USD_WRITEBACK) != desired:
            settings.set(USD_WRITEBACK, desired)
        result = original(*args, **kwargs)
        stats["simulation_groups"] += 1
        stats["published_groups" if final else "suppressed_groups"] += 1
        pending = not final
        return result

    env._usd_writeback_active = True
    env.sim_step = step
    try:
        yield
        if pending:
            raise RuntimeError("Control queue ended without its final USD publish")
    except BaseException:
        stats["exceptions"] += 1
        raise
    finally:
        if had_override:
            env.sim_step = original
        else:
            del env.sim_step
        settings.set(USD_WRITEBACK, saved)
        env._usd_writeback_active = False
        # Never issue an unpaired fetch or transformation update on failure.
        # Existing evaluation recovery owns resetting the failed scene.
