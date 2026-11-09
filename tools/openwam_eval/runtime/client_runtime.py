"""Explicit PT16 client instrumentation; native actions and scoring stay unchanged."""

import functools
import json
import os
from pathlib import Path
import time

import numpy as np


def instrument_factory(factory):
    @functools.wraps(factory)
    def create(*args, **kwargs):
        env = factory(*args, **kwargs)
        manager = env.seed_manager
        ids = json.loads(os.environ["OW_LAYOUT_IDS"])
        assert len(ids) == len(set(ids)) and all(i in manager.seed_info for i in ids)
        manager.seed_list = [i for i in ids if i in manager.seed_list]
        manager.st_idx = manager.idx = 0
        manager.ed_idx = len(manager.seed_list)
        _record_metrics(env)
        _verify_runtime(env)
        _install_optimizations(env)
        return env

    return create


def _record_metrics(env):
    out = Path(os.environ["OW_METRICS"])

    def record(data):
        data.update(t=time.time(), mono=time.perf_counter())
        with out.open("a") as f:
            f.write(json.dumps(data) + "\n")

    record(dict(kind="created", num_envs=env.num_envs, tensor_device=str(env.device), use_fabric=bool(env.use_fabric)))
    for name in ["reset", "get_obs_batch", "take_action_batch", "run_eval"]:
        old = getattr(env, name)

        def wrap(*args, _old=old, _name=name, **kwargs):
            before = list(env.take_action_cnt)
            start = time.perf_counter()
            if _name == "run_eval":
                record(dict(kind="run_start"))
            try:
                return _old(*args, **kwargs)
            finally:
                data = dict(kind=_name, seconds=time.perf_counter() - start)
                if _name == "reset":
                    context = getattr(getattr(env, "sim", None), "sim", None)
                    data.update(
                        tensor_device=str(env.device),
                        use_fabric=bool(env.use_fabric),
                        simulation_device=str(context.device) if context is not None else None,
                    )
                if _name == "take_action_batch":
                    data.update(
                        steps=sum(max(0, b - a) for a, b in zip(before, env.take_action_cnt)),
                        counts=list(env.take_action_cnt),
                    )
                if _name == "run_eval":
                    data.update(
                        counts=list(env.take_action_cnt),
                        success=env.success_nums,
                        fail=env.fail_nums,
                        score=env.total_score,
                    )
                record(data)

        setattr(env, name, wrap)
    audit_original = env.get_obs_batch
    audit_sequence = {}

    def audited_observation(*args, **kwargs):
        observations = audit_original(*args, **kwargs)
        for obs in observations:
            idx = int(obs["env_idx"])
            layout = int(env.env_seeds[idx])
            seq = audit_sequence.get(layout, 0)
            audit_sequence[layout] = seq + 1
            obs["_input_audit"] = dict(
                run_id=os.environ["ROBODOJO_RUN_ID"],
                seed=int(os.environ["OW_SEED"]),
                layout_id=layout,
                worker=int(os.environ["OW_WORKER"]),
                env_idx=idx,
                obs_sequence=seq,
            )
        return observations

    env.get_obs_batch = audited_observation


def _verify_runtime(env):
    import carb

    settings = carb.settings.get_settings()
    root = Path(os.environ["RENDER_PROBE_OUT"])
    root.mkdir(exist_ok=True)
    from omegaconf import open_dict

    with open_dict(env.sim_config):
        env.sim_config.defer_usd_writeback = True
        env.sim_config.cache_tic_geometry = hasattr(env, "check_chessboard_state")
        env.sim_config.capture_eval_warmup_observations = False
    if hasattr(env, "check_chessboard_state"):
        original_check = env.check_chessboard_state
        comparisons = []

        def check(env_idx):
            if len(comparisons) >= 50:
                return original_check(env_idx)
            before = env.sim.sim.current_time
            t = time.perf_counter()
            expected = env._check_chessboard_state(env_idx)
            native_seconds = time.perf_counter() - t
            t = time.perf_counter()
            actual = original_check(env_idx)
            optimized_seconds = time.perf_counter() - t
            assert actual == expected, f"Board query mismatch env{env_idx}: {expected} != {actual}"
            assert env.sim.sim.current_time == before
            comparisons.append(
                dict(
                    env=env_idx,
                    physics_time=before,
                    result=actual,
                    native_seconds=native_seconds,
                    optimized_seconds=optimized_seconds,
                )
            )
            (root / "GEOMETRY_SHADOW_CHECKS.json").write_text(
                json.dumps(
                    dict(comparisons=comparisons, matched=len(comparisons), all_equal=True, physics_unchanged=True),
                    indent=2,
                )
            )
            return actual

        env.check_chessboard_state = check
    original_action = env.take_action_batch
    action_calls = 0
    physics_calls = 0
    ik_calls = 0
    original_step = env.sim_step

    def step(*args, **kwargs):
        nonlocal physics_calls
        physics_calls += 1
        return original_step(*args, **kwargs)

    env.sim_step = step
    original_setup = env.setup_scene

    def setup(*args, **kwargs):
        context = env.sim.sim
        before = context.current_time
        before_steps = physics_calls
        start = time.time()
        manager = env.obs_manager
        capture = manager.get_obs
        capture_owned = "get_obs" in vars(manager)
        render = context.render
        render_owned = "render" in vars(context)
        captures = 0
        renders = 0

        def observe_warmup(*a, **k):
            nonlocal captures
            captures += 1
            return capture(*a, **k)

        def render_warmup(*a, **k):
            nonlocal renders
            renders += 1
            return render(*a, **k)

        manager.get_obs = observe_warmup
        context.render = render_warmup
        try:
            result = original_setup(*args, **kwargs)
            assert captures == 0, "Discarded warmup RGB captures remain active"
            assert physics_calls - before_steps == 500, "Native300stability+200settling env-step count changed"
            total_steps = round((context.current_time - before) / context.get_physics_dt())
            if hasattr(env, "check_chessboard_state"):
                assert total_steps == 540, "TIC native placement/stability/settling physics count changed"
            else:
                assert total_steps >= 500, "Native setup physics time is shorter than its500 recorded env steps"
            return result
        finally:
            if capture_owned:
                manager.get_obs = capture
            else:
                del manager.get_obs
            if render_owned:
                context.render = render
            else:
                del context.render
            (root / "STARTUP_WARMUP_VERIFIED.json").write_text(
                json.dumps(
                    dict(
                        time=time.time(),
                        seconds=time.time() - start,
                        physics_time_before=before,
                        physics_time_after=context.current_time,
                        env_physics_groups=physics_calls - before_steps,
                        total_physics_steps=round((context.current_time - before) / context.get_physics_dt()),
                        discarded_rgb_captures=captures,
                        sdk_render_calls=renders,
                    ),
                    indent=2,
                )
            )

    env.setup_scene = setup

    original_ik = env.robot_manager.solve_ik_batch

    def ik(*args, **kwargs):
        nonlocal ik_calls
        ik_calls += 1
        return original_ik(*args, **kwargs)

    env.robot_manager.solve_ik_batch = ik

    def action(*args, **kwargs):
        nonlocal action_calls, physics_calls, ik_calls
        if action_calls == 0:
            physics_calls = ik_calls = 0
        result = original_action(*args, **kwargs)
        action_calls += 1
        (root / "OPTIMIZATION_LATEST.json").write_text(
            json.dumps(
                dict(
                    time=time.time(),
                    action_calls=action_calls,
                    physics_calls=physics_calls,
                    ik_batch_calls=ik_calls,
                    usd_writeback=getattr(env, "usd_writeback_stats", None),
                    geometry=getattr(env.reward_manager.func_parser.layout_manager, "layout_snapshot_stats", None),
                    counts=list(env.take_action_cnt),
                ),
                indent=2,
            )
        )
        return result

    env.take_action_batch = action
    original_reset = env.reset
    original_obs = env.get_obs_batch
    seen = False

    def reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        context = env.sim.sim
        before = context.current_time
        settings.set_string("/rtx/rendermode", "PathTracing")
        settings.set_int("/rtx/pathtracing/spp", 16)
        settings.set_int("/rtx/pathtracing/totalSpp", 16)
        settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", True)
        for _ in range(8):
            context.render()
        assert context.current_time == before, "Physics changed during renderer warmup"
        assert settings.get("/rtx/rendermode") == "PathTracing", "Renderer mode did not persist"
        (root / "MODE_VERIFIED.json").write_text(
            json.dumps(
                {
                    "time": time.time(),
                    "protocol": "xstar-pt16-rigid-v1",
                    "simulation_device": str(context.device),
                    "use_fabric": bool(env.use_fabric),
                    "mode": settings.get("/rtx/rendermode"),
                    "spp": settings.get("/rtx/pathtracing/spp"),
                    "denoiser": settings.get("/rtx/pathtracing/optixDenoiser/enabled"),
                    "warmup_renders": 8,
                    "physics_time_before": before,
                    "physics_time_after": context.current_time,
                },
                indent=2,
            )
        )
        return result

    def observe(*args, **kwargs):
        nonlocal seen
        assert settings.get("/rtx/rendermode") == "PathTracing", "Renderer reverted before capture"
        result = original_obs(*args, **kwargs)
        assert settings.get("/rtx/rendermode") == "PathTracing", "Renderer reverted during capture"
        if not seen:
            all_images = []
            for item in result:
                for camera in ["cam_head", "cam_left_wrist", "cam_right_wrist"]:
                    arr = np.asarray(item["vision"][camera]["color"])
                    row = dict(
                        env=int(item["env_idx"]),
                        camera=camera,
                        shape=list(arr.shape),
                        mean=float(arr.mean()),
                        std=float(arr.std()),
                        finite=bool(np.isfinite(arr).all()),
                    )
                    all_images.append(row)
            (root / "ALL_FIRST_OBSERVATION.json").write_text(
                json.dumps(dict(time=time.time(), batch=len(result), stats=all_images), indent=2)
            )
            assert all(
                x["shape"] == [480, 640, 3] and x["finite"] and x["std"] > 5 and 20 < x["mean"] < 240
                for x in all_images
            )
            seen = True
        return result

    env.reset = reset
    env.get_obs_batch = observe


def _install_optimizations(env):
    from counter_delivery import capture_fence
    from gripper_snapshot import install

    import env.camera_manager.capture.render_sync as render_sync
    from env.environment.task_env import TaskEnv

    env._stream_vision = lambda *args, **kwargs: None
    os.environ.setdefault("PERF_WORKER_OUT", os.environ["RENDER_PROBE_OUT"])
    install(env, enabled=getattr(type(env).__mro__[1], "step", None) is TaskEnv.step, shadow_epochs=64)
    native_fence = render_sync.wait_for_latest_cameras

    def fence(context, *args, **kwargs):
        if kwargs.get("min_passes", 1) != 2:
            return native_fence(context, *args, **kwargs)
        return capture_fence(context, native_fence)

    render_sync.wait_for_latest_cameras = fence
