"""Exercise the PR9 fallback without importing a simulator or allocating a GPU."""
import ast
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class RewardPoseModeTests(unittest.TestCase):
    def test_disabled_mode_uses_original_poses_and_restores_nested_snapshot(self):
        path = Path(__file__).parents[1] / "env/scene_manager/layout_manager.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LayoutManager")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "cpu_poses")
        usd = SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: None))
        geom = SimpleNamespace(XformCache=lambda *_: None)
        module = SimpleNamespace(Usd=usd, UsdGeom=geom)
        ns = {"contextmanager": contextmanager, "cpu_snapshot": lambda values: dict(values)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        old_poses, old_bbox = {"prior": True}, {"prior_bbox": True}
        calls = []
        manager = SimpleNamespace(
            _batch_reward_poses=False, _cpu_poses=old_poses,
            _projected_bbox_cache=old_bbox,
            instance_type_by_env=[{"box": "rigid", "cloth": "garment"}],
            scene_manager=SimpleNamespace(env_origins=[1]),
            _instance_pose=lambda idx, name: (calls.append((idx, name)) or ([2], [3])),
            get_scene_object=lambda *_: self.fail("Disabled mode must not query world poses"),
        )
        with patch.dict(sys.modules, {"pxr": module}):
            with ns["cpu_poses"](manager):
                outer = manager._cpu_poses
                self.assertEqual(outer[(0, "box", "pos")], [2])
                self.assertEqual(outer[(0, "cloth", "rot")], [3])
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    with ns["cpu_poses"](manager):
                        raise RuntimeError("stop")
                self.assertIs(manager._cpu_poses, outer)
                self.assertEqual(manager._projected_bbox_cache, {})
        self.assertIs(manager._cpu_poses, old_poses)
        self.assertIs(manager._projected_bbox_cache, old_bbox)
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
