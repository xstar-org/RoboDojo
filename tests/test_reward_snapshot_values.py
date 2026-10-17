"""Compare scoped robot reads against the untouched original read branches."""
import ast
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(os.environ.get('XSTAR_REWARD_TEST_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from utils.cpu_snapshot import cpu_snapshot


class SnapshotValueTests(unittest.TestCase):
    def test_coalesced_transfer_preserves_dtypes_shapes_and_source_ownership(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        source = torch.arange(48, dtype=torch.float32, device=device).reshape(4, 12)
        fields = {i: source[i, ::2] for i in range(4)}
        fields['count'] = torch.tensor([3, 2], device=device)
        fields['empty'] = torch.empty((0, 3), device=device)
        with patch('utils.cpu_snapshot.torch.cat', wraps=torch.cat) as cat:
            actual = cpu_snapshot(fields)
        self.assertEqual(cat.call_count, 2)  # float and integer groups
        for k, value in fields.items():
            np.testing.assert_array_equal(actual[k], value.cpu().numpy())
        source[:] = -1
        np.testing.assert_array_equal(actual[0], np.arange(0, 12, 2))
        self.assertEqual(actual['count'].dtype, np.int64)
        self.assertEqual(actual['empty'].shape, (0, 3))

    def test_robot_cached_getters_equal_original_for_subsets_and_relative_poses(self):
        path = ROOT/'env/robot_manager/robot_manager.py'
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        names = {'cpu_state', '_host_state', '_select_state', 'get_joint',
                 'get_end_effector_real_val', 'get_link_pose'}
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = dict(contextmanager=contextmanager, deepcopy=deepcopy,
                         cpu_snapshot=cpu_snapshot, np=np, torch=torch)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
        m = namespace['RobotManager']()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        robot = SimpleNamespace(arm_joint_indices=[0, 2], gripper_joint_indices=[1], robot_name='fixture')
        key = SimpleNamespace(body_names=['root', 'tool'], data=SimpleNamespace(
            joint_pos=torch.arange(12, dtype=torch.float32, device=device).reshape(4, 3),
            body_link_pose_w=torch.arange(56, dtype=torch.float32, device=device).reshape(4, 2, 7)))
        m._cpu_state = None
        m.robot_list, m.robot_key, m.num_envs = [robot], [key], 4
        m.scene = SimpleNamespace(env_origins=torch.arange(12, dtype=torch.float32, device=device).reshape(4, 3))
        for selection in (None, [], [3, 1]):
            for relative in (False, True):
                def read():
                    return (m.get_joint(robot, selection), m.get_end_effector_real_val(robot, selection),
                            m.get_link_pose(robot, 'tool', selection, relative))
                expected = read()
                with m.cpu_state(poses=True):
                    actual = read()
                    for a, b in zip(actual, expected):
                        self.assertEqual(a.keys(), b.keys())
                        for k in a:
                            if b[k] is None:
                                self.assertIsNone(a[k])
                            else:
                                np.testing.assert_array_equal(a[k], b[k])
                    for result in actual:
                        for v in result.values():
                            if v is not None: v[:] = -999
                    for a, b in zip(read(), expected):
                        for k in a:
                            if b[k] is not None: np.testing.assert_array_equal(a[k], b[k])
                self.assertIsNone(m._cpu_state)


if __name__ == '__main__':
    unittest.main()
