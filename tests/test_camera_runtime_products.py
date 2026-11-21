"""Exercise actual camera-manager methods with SDK resource doubles."""
import ast
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import List
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "env/camera_manager/capture/tiled_capture_manager.py"


class Config(dict):
    __getattr__ = dict.__getitem__


def manager(sim6, backend="tiled"):
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
    cfg = Config(render_product_backend=backend,
                 annotator={"common": {"enabled": True, "rgb_capture": {"type": "rgb"}}})
    instance = namespace["Manager"](2, cfg, camera_manager, "cpu")
    instance.num_cams = 3
    instance._release_tiled_cameras = lambda: instance.tiled_cameras.clear()
    context = SimpleNamespace(current_time=1)
    instance.sim = SimpleNamespace(sim=context)
    return instance, views, events, context


class CameraRuntimeTests(unittest.TestCase):
    def test_camera_creation_skips_owner_product_update_option(self):
        source = SOURCE.parent.parent / "camera_manager.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        camera_manager = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        create_cameras = next(node for node in camera_manager.body
                              if isinstance(node, ast.FunctionDef) and node.name == "create_cameras")
        original = Config(isolated_owner_products_only=True, cam_head={"camera": {"mount_link": "link"}})
        manager = SimpleNamespace(num_envs=1, init_config=original, cameras=[], cameras_xform_path=[],
                                  cameras_xform=[], action_noise=[], camera_names=[])
        lookup = Mock(side_effect=RuntimeError("camera creation reached"))
        def create_config(value):
            return Config({key: create_config(item) if isinstance(item, dict) else item for key, item in value.items()})

        namespace = dict(deepcopy=deepcopy, OmegaConf=SimpleNamespace(to_container=lambda c, **kw: dict(c),
                                                                   create=create_config),
                         find_unique_string_name=lookup)
        exec(compile(ast.Module(body=[create_cameras], type_ignores=[]), str(source), "exec"), namespace)
        with self.assertRaisesRegex(RuntimeError, "camera creation reached"):
            namespace["create_cameras"](manager)
        self.assertEqual(manager.camera_names, [["cam_head"]])
        self.assertTrue(original.isolated_owner_products_only)

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

    def test_isolated_capture_preserves_order_and_restores_visibility_on_error(self):
        m, _, events, context = manager(True, "independent_isolated")
        active = []
        product_updates = [True] * 6
        m._isolate_product_updates = True
        m.tiled_render_products = [
            SimpleNamespace(hydra_texture=SimpleNamespace(
                set_updates_enabled=lambda enabled, index=index: product_updates.__setitem__(index, enabled)
            ))
            for index in range(6)
        ]

        @contextmanager
        def only(owner):
            active.append(owner)
            try:
                yield
            finally:
                active.pop()

        def read(env_ids, cam_ids):
            self.assertEqual(active, env_ids)
            self.assertEqual(product_updates, [index % 2 == env_ids[0] for index in range(6)])
            return [{"rgb": [(camera, env_ids[0])]} for camera in cam_ids]

        visibility = ModuleType("env.camera_manager.capture.environment_visibility")
        visibility.EnvironmentVisibility = lambda *args: SimpleNamespace(only=only)
        sync = ModuleType("env.camera_manager.capture.render_sync")
        sync.wait_for_latest_cameras = lambda ctx, **kw: events.append((ctx, kw, list(active)))
        usd = ModuleType("omni.usd")
        usd.get_context = lambda: SimpleNamespace(get_stage=lambda: "stage")
        omni = ModuleType("omni")
        omni.usd = usd
        modules = {module.__name__: module for module in (omni, usd, visibility, sync)}
        with patch.dict(sys.modules, modules):
            m._read_camera = read
            result = m.step(env_ids=[1, 0], cam_ids=[2, 0])
            self.assertEqual(result, [{"rgb": [(2, 1), (2, 0)]}, {"rgb": [(0, 1), (0, 0)]}])
            self.assertEqual(events, [(context, {"min_passes": 2}, [1]),
                                      (context, {"min_passes": 2}, [0])])
            self.assertEqual(active, [])
            self.assertEqual(product_updates, [True] * 6)

            def fail(**kwargs):
                raise RuntimeError("annotator error")

            m._read_camera = fail
            with self.assertRaisesRegex(RuntimeError, "annotator error"):
                m.step(env_ids=[1], cam_ids=[2])
            self.assertEqual(active, [])
            self.assertEqual(product_updates, [True] * 6)

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
