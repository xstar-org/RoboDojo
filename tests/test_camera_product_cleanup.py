"""Exercise the actual cleanup methods without importing the Isaac runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1] / "env/camera_manager/capture"


def load_method(filename, class_name, method_name):
    path = ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


cleanup = load_method("camera_view.py", "CameraView", "_clean_up_tiled_sensor")
release = load_method("tiled_capture_manager.py", "TiledCaptureManager", "_release_tiled_cameras")


class ProductCleanupTests(unittest.TestCase):
    def make_view(self, events, label, detach_error=False):
        def detach(paths):
            events.append((label, "detach", paths))
            if detach_error:
                raise RuntimeError("detachment failed")

        product = SimpleNamespace(path=label, destroy=lambda: events.append((label, "destroy")))
        view = SimpleNamespace(_render_product=product,
                               _annotators={"rgba": SimpleNamespace(detach=detach)})
        view._clean_up_tiled_sensor = lambda: cleanup(view)
        return view

    def test_partial_constructor_and_repeated_cleanup_are_safe(self):
        cleanup(SimpleNamespace())
        events = []
        view = self.make_view(events, "owned")
        cleanup(view)
        cleanup(view)
        self.assertEqual(events, [("owned", "detach", ["owned"]), ("owned", "destroy")])
        self.assertIsNone(view._render_product)
        self.assertEqual(view._annotators, {})

    def test_every_annotator_is_detached_before_product_destruction(self):
        events = []
        view = self.make_view(events, "owned")
        view._annotators["depth"] = SimpleNamespace(detach=lambda paths: events.append(("depth", paths)))
        cleanup(view)
        self.assertEqual(events[-1], ("owned", "destroy"))
        self.assertEqual(len(events), 3)

    def test_sdk_error_still_destroys_product_and_clears_ownership(self):
        events = []
        view = self.make_view(events, "owned", detach_error=True)
        with self.assertRaisesRegex(RuntimeError, "detachment failed"):
            cleanup(view)
        cleanup(view)
        self.assertEqual(events[-1], ("owned", "destroy"))
        self.assertEqual(len(events), 2)
        self.assertIsNone(view._render_product)

    def test_manager_releases_all_views_and_is_idempotent(self):
        events = []
        views = [self.make_view(events, name) for name in ["a", "b"]]
        manager = SimpleNamespace(tiled_cameras=views, tiled_render_products=[v._render_product for v in views])
        release(manager)
        release(manager)
        self.assertEqual(manager.tiled_cameras, [])
        self.assertEqual(manager.tiled_render_products, [])
        self.assertEqual([event for event in events if event[1] == "destroy"],
                         [("a", "destroy"), ("b", "destroy")])

    def test_manager_error_does_not_leave_other_views_live(self):
        events = []
        a = self.make_view(events, "a", detach_error=True)
        b = self.make_view(events, "b")
        manager = SimpleNamespace(tiled_cameras=[a, b], tiled_render_products=[a._render_product, b._render_product])
        with self.assertRaisesRegex(RuntimeError, "detachment failed"):
            release(manager)
        self.assertIn(("b", "destroy"), events)
        self.assertEqual(manager.tiled_cameras, [])
        self.assertEqual(manager.tiled_render_products, [])


if __name__ == "__main__":
    unittest.main()
