import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from utils.cpu_snapshot import cpu_snapshot
from utils.transformer import calc_polygon


class RewardFootprintCacheTests(unittest.TestCase):
    def make_manager(self):
        path = Path(__file__).parents[1] / "env/scene_manager/layout_manager.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LayoutManager")
        keep = {"cpu_poses", "get_instance_pose", "get_instance_projected_bbox"}
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in keep]
        project = Mock(wraps=calc_polygon)
        namespace = {"np": np, "contextmanager": contextmanager, "cpu_snapshot": cpu_snapshot, "calc_polygon": project}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        manager = namespace["LayoutManager"]()
        manager._cpu_poses = manager._projected_bbox_cache = None
        manager.instance_type_by_env = [{"box": "rigid"}]
        manager.scene_manager = SimpleNamespace(env_origins=np.zeros((1, 3)))
        manager.position = np.zeros(3)
        manager._instance_pose = lambda *args: (manager.position, np.array([1.0, 0.0, 0.0, 0.0]))
        manager.get_instance_bbox_vertices = lambda **kwargs: np.array(
            [[x, y, z] for x in [-1.0, 1.0] for y in [-1.0, 1.0] for z in [-1.0, 1.0]]
        )
        return manager, project

    def test_reuses_identical_geometry_within_one_snapshot(self):
        manager, project = self.make_manager()
        with manager.cpu_poses():
            for _ in range(50):
                polygon, lo, hi = manager.get_instance_projected_bbox("box", 0)
                self.assertEqual(polygon.area, 4.0)
                self.assertEqual((lo, hi), (-1.0, 1.0))
        self.assertEqual(project.call_count, 1)

    def test_exception_clears_cache_and_next_state_is_fresh(self):
        manager, project = self.make_manager()
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with manager.cpu_poses():
                manager.get_instance_projected_bbox("box", 0)
                raise RuntimeError("stop")
        self.assertIsNone(manager._cpu_poses)
        self.assertIsNone(manager._projected_bbox_cache)
        manager.position[2] = 10.0
        with manager.cpu_poses():
            _, lo, hi = manager.get_instance_projected_bbox("box", 0)
            self.assertEqual((lo, hi), (9.0, 11.0))
        self.assertEqual(project.call_count, 2)

    def test_outside_scope_never_reuses_stale_geometry(self):
        manager, project = self.make_manager()
        manager.get_instance_projected_bbox("box", 0)
        manager.position[2] = 3.0
        _, lo, hi = manager.get_instance_projected_bbox("box", 0)
        self.assertEqual((lo, hi), (2.0, 4.0))
        self.assertEqual(project.call_count, 2)


if __name__ == "__main__":
    unittest.main()
