import contextvars
import functools
import json
import os
from pathlib import Path
import time

ROOT = Path(os.environ["PERF_ROOT"])
ACTIVE = contextvars.ContextVar("performance_worker", default="unknown")


def record(kind, **row):
    with (ROOT / "policy-metrics.jsonl").open("a") as f:
        f.write(json.dumps(dict(kind=kind, t=time.time(), worker=ACTIVE.get(), **row)) + "\n")


def install(model):
    if getattr(model, "_performance_instrumented", False):
        return
    model._performance_instrumented = True
    engine = model._engine
    old = engine.generate_batch
    architecture = engine.architecture
    real = architecture.generate_batch

    @functools.wraps(real)
    def architecture_batch(*args, **kwargs):
        samples = args[1] if len(args) > 1 else kwargs["samples"]
        proprio = kwargs.get("proprio")
        record(
            "architecture_batch", batch=len(samples), proprio_shape=list(proprio.shape) if proprio is not None else None
        )
        return real(*args, **kwargs)

    architecture.generate_batch = architecture_batch
    backbone = getattr(architecture, "action_backbone", None)
    method = next((n for n in ["predict_velocity", "forward"] if callable(getattr(backbone, n, None))), None)
    if method:
        original = getattr(backbone, method)
        seen = [False]

        @functools.wraps(original)
        def tensor_call(*args, **kwargs):
            if not seen[0]:
                shapes = {str(i): list(x.shape) for i, x in enumerate(args) if hasattr(x, "shape")}
                shapes.update({k: list(x.shape) for k, x in kwargs.items() if hasattr(x, "shape")})
                record("first_backbone_tensors", method=method, shapes=shapes)
                seen[0] = True
            return original(*args, **kwargs)

        setattr(backbone, method, tensor_call)

    @functools.wraps(old)
    def generate(conditions):
        assert 1 <= len(conditions) <= 32
        start = time.perf_counter()
        result = old(conditions)
        record(
            "engine_batch",
            batch=len(conditions),
            action_shape=list(result["actions"].shape),
            seconds=time.perf_counter() - start,
        )
        p = ROOT / "ROI_STARTED.json"
        if not p.exists():
            q = p.with_suffix(".tmp")
            q.write_text(
                json.dumps(
                    dict(
                        time=time.time(),
                        batch=len(conditions),
                        scope="Scored run first real inference; not a benchmark synchronization boundary",
                    )
                )
            )
            q.replace(p)
        return result

    old_joint = architecture._run_joint_denoising

    @functools.wraps(old_joint)
    def joint(*args, **kwargs):
        inputs = args[1]
        actions = args[2]
        record(
            "joint_denoising_tensors",
            action_shape=list(actions.shape),
            video_shape=list(inputs["latents"].shape),
            proprio_shape=list(inputs["proprio"].shape),
        )
        return old_joint(*args, **kwargs)

    architecture._run_joint_denoising = joint
    engine.generate_batch = generate
