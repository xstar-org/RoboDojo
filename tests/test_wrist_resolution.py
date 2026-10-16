"""Tests for wrist camera resolution override. No Isaac imports required."""
import ast
import importlib.util
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
MGR = REPO / "env" / "camera_manager" / "camera_manager.py"


def load_helper():
    """Import only the pure helper by exec'ing the module top section."""
    src = MGR.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("REAL_MAP", "VISUAL_MAP") for t in node.targets
        ):
            wanted.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_camera_resolution":
            wanted.append(node)
    namespace = {}
    # Provide stubs so map assignments that reference template names still work.
    for name in ("GEMINI_345LG", "THIRD_VIEW", "D435", "LARGE_D435", "PINHOLE"):
        namespace[name] = {"resolution": (640, 480)}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(MGR), "exec"), namespace)
    return namespace["resolve_camera_resolution"]


class WristResolutionTests(unittest.TestCase):
    def test_override_applies_only_to_wrist_cameras(self):
        fn = load_helper()
        self.assertEqual(fn("cam_left_wrist", (640, 480), (160, 120)), (160, 120))
        self.assertEqual(fn("cam_right_wrist", (640, 480), (160, 120)), (160, 120))
        self.assertEqual(fn("cam_head", (640, 480), (160, 120)), (640, 480))

    def test_none_override_keeps_template_resolution(self):
        fn = load_helper()
        self.assertEqual(fn("cam_left_wrist", (640, 480), None), (640, 480))
        self.assertEqual(fn("cam_head", (640, 480), None), (640, 480))

    def test_manager_reads_key_from_config(self):
        """AST check: initialize() pops the key and all creation sites route through the helper."""
        tree = ast.parse(MGR.read_text(encoding="utf-8"))
        sources = [
            ast.dump(node) for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", "") == "resolve_camera_resolution"
        ]
        self.assertEqual(len(sources), 3, f"expected 3 resolved creation sites, got {len(sources)}")
        self.assertIn("wrist_camera_resolution", MGR.read_text(encoding="utf-8"))

    def test_yaml_exposes_the_override(self):
        text = (REPO / "env_cfg" / "camera" / "camera_config.yml").read_text(encoding="utf-8")
        self.assertIn("wrist_camera_resolution: [640, 480]", text)


if __name__ == "__main__":
    unittest.main()
