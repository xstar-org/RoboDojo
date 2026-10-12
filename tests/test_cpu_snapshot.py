import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from utils.cpu_snapshot import cpu_snapshot


class CpuSnapshotTests(unittest.TestCase):
    def test_velocity_reset_uses_combined_gpu_api(self):
        path = Path(__file__).parents[1] / "env/scene_manager/objects/rigid.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_default_velocities")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        view = Mock()
        view._device = "cpu"
        view._backend_utils.convert.side_effect = lambda x, **kwargs: torch.tensor(x)
        obj = SimpleNamespace(
            _default_linear_velocity=[0.0, 0.0, 0.0],
            _default_angular_velocity=[0.0, 0.0, 0.0],
            _rigid_prim_view=view,
            set_linear_velocity=Mock(), set_angular_velocity=Mock(),
        )
        namespace["_apply_default_velocities"](obj)
        view.set_velocities.assert_called_once()
        self.assertEqual(tuple(view.set_velocities.call_args.args[0].shape), (1, 6))
        self.assertEqual(view.set_velocities.call_args.args[0].count_nonzero(), 0)
        obj.set_linear_velocity.assert_not_called()
        obj.set_angular_velocity.assert_not_called()

    def test_shapes_dtypes_and_ownership(self):
        source = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        result = cpu_snapshot({"pose": source, "count": torch.tensor([2]), "array": np.ones(2)})
        source.fill_(99)
        np.testing.assert_array_equal(result["pose"], np.arange(30).reshape(10, 3))
        self.assertEqual(result["count"].dtype, np.int64)
        self.assertEqual(result["pose"].shape, (10, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_coalesces_environments_and_preserves_values(self):
        source = torch.arange(150, device="cuda", dtype=torch.float32).reshape(10, 15)
        with patch("utils.cpu_snapshot.torch.cat", wraps=torch.cat) as cat:
            result = cpu_snapshot({i: source[i] for i in range(10)})
        self.assertEqual(cat.call_count, 1)
        np.testing.assert_array_equal(np.stack(list(result.values())), source.cpu().numpy())

    def test_robot_scope_refreshes_and_clears_on_exception(self):
        # Exercise actual manager methods without loading the Isaac application.
        path = Path(__file__).parents[1] / "env/robot_manager/robot_manager.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        names = {"cpu_state", "_host_state", "_select_state", "get_joint"}
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {"contextmanager": contextmanager, "cpu_snapshot": cpu_snapshot}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        manager = namespace["RobotManager"]()
        robot = SimpleNamespace(arm_joint_indices=[0, 2])
        key = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.arange(12).reshape(4, 3)))
        manager._cpu_state = None
        manager.num_envs = 4
        manager.robot_list = [robot]
        manager.robot_key = [key]
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with manager.cpu_state():
                key.data.joint_pos.fill_(77)
                np.testing.assert_array_equal(manager.get_joint(robot, [1])[1], [3, 5])
                self.assertIsNone(manager.get_joint(robot, [1])[0])
                raise RuntimeError("stop")
        self.assertIsNone(manager._cpu_state)
        with manager.cpu_state():
            np.testing.assert_array_equal(manager.get_joint(robot)[1], [77, 77])


if __name__ == "__main__":
    unittest.main()
