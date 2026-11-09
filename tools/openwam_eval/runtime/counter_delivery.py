"""Keep two native SDG deliveries; bound PT accumulation to one observation."""

import json
import os
from pathlib import Path
import time

SEQUENCE = 0


def capture_fence(ctx, native_fence):
    global SEQUENCE
    import carb.settings

    s = carb.settings.get_settings()
    flag = "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges"
    counter = "/rtx/externalFrameCounter"
    saved = {k: s.get(k) for k in [flag, counter]}
    assert all(v is not None for v in saved.values()) or callable(getattr(s, "destroy_item", None))
    assert (
        s.get("/rtx/rendermode") == "PathTracing"
        and s.get("/rtx/pathtracing/spp") == 16
        and s.get("/rtx/pathtracing/totalSpp") == 16
        and s.get("/rtx/pathtracing/optixDenoiser/enabled")
    )
    SEQUENCE = max(SEQUENCE, int(saved[counter] or 0)) + 1
    before = ctx.current_time
    physical = []
    name = "perf_counter_freeze_" + str(os.getpid())
    ctx.add_physics_callback(name, lambda dt: physical.append(float(dt)))
    from env.camera_manager.capture.render_sync import dispatcher_time

    stamps = [dispatcher_time()]
    original_render = ctx.render

    def trace_render(*a, **kw):
        value = original_render(*a, **kw)
        stamps.append(dispatcher_time())
        return value

    ctx.render = trace_render
    started = time.perf_counter()
    try:
        s.set_bool(flag, True)
        s.set_int(counter, SEQUENCE)
        passes = native_fence(ctx, min_passes=2)
    finally:
        ctx.render = original_render
        # Native references run after restoring the original accumulation rules.
        for key, value in saved.items():
            if value is None:
                s.destroy_item(key)
            else:
                s.set(key, value)
        ctx.remove_physics_callback(name)
    restored = {k: s.get(k) for k in saved}
    delivered = sum(a is not None and b is not None and a != b for a, b in zip(stamps, stamps[1:]))
    row = dict(
        dispatcher_stamps=stamps,
        verified_deliveries=delivered,
        time=time.time(),
        seconds=time.perf_counter() - started,
        observation_counter=SEQUENCE,
        native_delivery_passes=passes,
        physics_before=before,
        physics_after=ctx.current_time,
        physics_callback_dts=physical,
        original=saved,
        restored=restored,
        spp=s.get("/rtx/pathtracing/spp"),
        total_spp=s.get("/rtx/pathtracing/totalSpp"),
        reference_rules_restored=saved == restored,
    )
    with (Path(os.environ["PERF_WORKER_OUT"]) / "counter-delivery.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    assert delivered >= 2 and passes >= 2 and not physical and before == ctx.current_time and saved == restored
    return passes
