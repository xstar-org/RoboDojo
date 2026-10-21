"""Exercise actual camera-manager methods with SDK resource doubles."""
import ast
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import List
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "env/camera_manager/capture/tiled_capture_manager.py"


class Config(dict):
    __getattr__ = dict.__getitem__


def manager(sim6):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
    views = []
    events = []

    class Pixels:
        def __init__(self, label): self.label = label
        def copy(self): return Pixels(self.label)

    def create_view(paths, **kw):
        index = len(views)
        product = SimpleNamespace(path=f"product{index}", hydra_texture=SimpleNamespace(set_updates_enabled=lambda _: None))
        def get_data(name, out=None):
            return [Pixels((index, env)) for env in range(2)], {}
        view = SimpleNamespace(_render_product=product, get_data=get_data)
        views.append(view)
        return view

    namespace = dict(List=List, DictConfig=object, CameraManager=object,
                     torch=SimpleNamespace(device=object), deepcopy=deepcopy,
                     is_sim6=lambda: sim6, CameraView=create_view,
                     OmegaConf=SimpleNamespace(to_container=lambda c, **kw: dict(c), create=Config))
    module = ast.Module(body=[ast.ClassDef(name="Manager", bases=[], keywords=[], body=methods,
                                          decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    cameras = [[SimpleNamespace(prim_path=f"/env{e}/cam{c}", _resolution=(640, 480)) for c in range(3)] for e in range(2)]
    camera_manager = SimpleNamespace(cameras=cameras, camera_names=[["head", "left", "right"] for _ in range(2)])
    cfg = Config(annotator={"common": {"enabled": True, "rgb_capture": {"type": "rgb"}}})
    instance = namespace["Manager"](2, cfg, camera_manager, "cpu")
    instance.num_cams = 3
    instance._release_tiled_cameras = lambda: instance.tiled_cameras.clear()
    context = SimpleNamespace(current_time=1)
    instance.sim = SimpleNamespace(sim=context)
    return instance, views, events, context


class CameraRuntimeTests(unittest.TestCase):
    def test_sim5_reuses_one_product_and_sim6_keeps_three_distinct_views(self):
        spec = SimpleNamespace(ANNOTATOR_SPEC={"rgb": {"channels": 4, "dtype": "uint8"}})
        warp = SimpleNamespace(zeros=lambda shape, **kw: shape)
        with patch.dict(sys.modules, {"env.camera_manager.capture.camera_view": spec, "warp": warp}):
            for sim6, count in [(False, 1), (True, 3)]:
                with self.subTest(sim6=sim6):
                    m, views, _, _ = manager(sim6)
                    m.init_cameras()
                    self.assertEqual(len(views), count)
                    self.assertEqual(len(m._output_buffers), count)
                    data = m._read_camera(env_ids=[1], cam_ids=[0, 1, 2])
                    labels = [row["rgb"][0]["data"].label for row in data]
                    expected = [(0, 1), (1, 1), (2, 1)] if sim6 else [(0, 1)] * 3
                    self.assertEqual(labels, expected)

    def test_sim6_waits_once_and_does_not_retarget_or_advance_physics(self):
        m, _, events, context = manager(True)
        m._read_camera = lambda **kw: ["head", "left", "right"]
        sync = ModuleType("env.camera_manager.capture.render_sync")
        sync.wait_for_latest_cameras = lambda ctx, **kw: events.append((ctx, kw))
        with patch.dict(sys.modules, {sync.__name__: sync}):
            self.assertEqual(m.step(env_ids=[0]), ["head", "left", "right"])
        self.assertEqual(events, [(context, {"min_passes": 2})])
        self.assertEqual(context.current_time, 1)

    def test_sim6_rejects_an_unexpected_physics_advance(self):
        m, _, _, context = manager(True)
        m._read_camera = lambda **kw: []
        sync = ModuleType("env.camera_manager.capture.render_sync")
        sync.wait_for_latest_cameras = lambda ctx, **kw: setattr(ctx, "current_time", 2)
        with patch.dict(sys.modules, {sync.__name__: sync}):
            with self.assertRaisesRegex(RuntimeError, "Physics advanced"):
                m.step()

    def test_sim5_still_retargets_and_waits_for_each_camera_type(self):
        m, _, events, _ = manager(False)
        m.camera_prim_paths = [[f"/env{e}/cam{c}" for c in range(3)] for e in range(2)]
        m.tiled_render_products = [SimpleNamespace(path="/product")]
        m._read_camera = lambda **kw: kw["cam_ids"]
        targets = []
        relationship = SimpleNamespace(SetTargets=targets.append)
        stage = SimpleNamespace(GetPrimAtPath=lambda _: SimpleNamespace(GetRelationship=lambda _: relationship),
                                GetSessionLayer=lambda: "session")
        usd = ModuleType("omni.usd")
        usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
        omni = ModuleType("omni")
        omni.usd = usd
        pxr = ModuleType("pxr")
        pxr.Usd = SimpleNamespace(EditContext=lambda *args: nullcontext())
        sync = ModuleType("env.camera_manager.capture.render_sync")
        sync.wait_for_latest_cameras = lambda ctx, **kw: events.append(kw)
        with patch.dict(sys.modules, {"omni": omni, "omni.usd": usd, "pxr": pxr, sync.__name__: sync}):
            self.assertEqual(m.step(), [0, 1, 2])
        self.assertEqual(targets, [[f"/env{e}/cam{c}" for e in range(2)] for c in range(3)])
        self.assertEqual(events, [{"min_passes": 2}] * 3)


if __name__ == "__main__":
    unittest.main()
