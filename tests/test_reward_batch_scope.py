"""Reward scope lifecycle without requiring an Isaac Sim installation."""
import ast
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def methods(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [m for c in tree.body if isinstance(c, ast.ClassDef)
             for m in c.body if isinstance(m, ast.FunctionDef) and m.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


class RewardScopeTests(unittest.TestCase):
    def test_layout_fresh_nested_exception_and_output_ownership(self):
        ns = methods("env/scene_manager/layout_manager.py", {"cpu_poses", "get_instance_pose"},
                     dict(contextmanager=contextmanager, np=np, deepcopy=deepcopy,
                          cpu_snapshot=lambda v: {k: np.array(x, copy=True) for k, x in v.items()}))
        position = np.array([1., 2., 3.])
        m = SimpleNamespace(_cpu_poses=None, _projected_bbox_cache=None,
                            instance_type_by_env=[{"cloth": "garment"}],
                            scene_manager=SimpleNamespace(env_origins=np.array([[10., 0., 0.]])),
                            _instance_pose=lambda *_: (position, np.array([1., 0., 0., 0.])))
        usd = SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: None))
        geom = SimpleNamespace(XformCache=lambda *_: None)
        with patch.dict(sys.modules, {"pxr": SimpleNamespace(Usd=usd, UsdGeom=geom)}):
            with ns["cpu_poses"](m):
                outer = m._cpu_poses
                pos, _ = ns["get_instance_pose"](m, 0, inst_name="cloth", relative=False)
                np.testing.assert_array_equal(pos, position)  # original cloth convention
                pos[:] = 99
                np.testing.assert_array_equal(outer[(0, "cloth", "pos")], position)
                position[:] = 4
                with self.assertRaisesRegex(RuntimeError, "nested"):
                    with ns["cpu_poses"](m):
                        np.testing.assert_array_equal(m._cpu_poses[(0, "cloth", "pos")], position)
                        raise RuntimeError("nested")
                self.assertIs(m._cpu_poses, outer)
            self.assertIsNone(m._cpu_poses)
            self.assertIsNone(m._projected_bbox_cache)
            with ns["cpu_poses"](m):
                np.testing.assert_array_equal(m._cpu_poses[(0, "cloth", "pos")], position)

    def test_disabled_mode_and_counter_checks_never_read_physics(self):
        ns = methods("env/reward_manager/reward_manager.py", {"_reward_snapshot", "get_reward"},
                     dict(contextmanager=contextmanager))
        m = SimpleNamespace(_batch_reward=False, env=None,
                            _snapshot_get_reward=lambda final: ("unchanged", final))
        m._reward_snapshot = lambda: ns["_reward_snapshot"](m)
        self.assertEqual(ns["get_reward"](m, True), ("unchanged", True))
        m._batch_reward = True
        self.assertEqual(ns["get_reward"](m, False), ("unchanged", False))

    def test_robot_scope_restores_after_failure_and_refreshes_next_call(self):
        ns = methods("env/robot_manager/robot_manager.py", {"cpu_state"},
                     dict(contextmanager=contextmanager,
                          cpu_snapshot=lambda v: {k: np.array(x, copy=True) for k, x in v.items()}))
        key = SimpleNamespace(data=SimpleNamespace(joint_pos=np.ones((2, 3)),
                                                   body_link_pose_w=np.ones((2, 4, 7))))
        m = SimpleNamespace(_cpu_state=None, robot_key=[key],
                            scene=SimpleNamespace(env_origins=np.zeros((2, 3))))
        with self.assertRaises(ValueError):
            with ns["cpu_state"](m, poses=True):
                self.assertFalse(np.shares_memory(m._cpu_state[(id(key), "joints")], key.data.joint_pos))
                with ns["cpu_state"](m, poses=True):
                    raise ValueError("predicate failed")
        self.assertIsNone(m._cpu_state)
        key.data.joint_pos[:] = 8
        with ns["cpu_state"](m, poses=True):
            np.testing.assert_array_equal(m._cpu_state[(id(key), "joints")], key.data.joint_pos)


if __name__ == "__main__":
    unittest.main()
