"""Exercise the evaluator's CPU target path without loading Isaac Sim."""
import ast
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from env.robot_manager.control_manager import ControlManager, MetaControl


def load_robot_control_methods():
    source = Path(__file__).resolve().parents[1] / "env/robot_manager/robot_manager.py"
    tree = ast.parse(source.read_text())
    manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RobotManager")
    names = {
        "get_end_effector_real_val", "use_batched_cpu_controls", "batch_control_gripper_reads",
        "_dispatch_batched_cpu_controls", "control_robot",
    }
    manager.body = [node for node in manager.body if isinstance(node, ast.FunctionDef) and node.name in names]
    environment = dict(np=np, torch=torch, contextmanager=contextmanager, deepcopy=deepcopy, MetaControl=MetaControl)
    exec(compile(ast.Module(body=[manager], type_ignores=[]), str(source), "exec"), environment)
    return environment["RobotManager"]


RobotManager = load_robot_control_methods()


class Articulation:
    def __init__(self, joints):
        self.data = SimpleNamespace(joint_pos=joints)
        self.targets = []

    def set_joint_position_target(self, values, joint_ids, env_ids):
        self.targets.append(("position", list(joint_ids), env_ids.clone(), values.clone()))

    def set_joint_velocity_target(self, values, joint_ids, env_ids):
        self.targets.append(("velocity", list(joint_ids), env_ids.clone(), values.clone()))


def make_manager(fast, count=4):
    manager = RobotManager()
    manager.batch_cpu_controls = fast
    manager._control_gripper_snapshot = None
    manager.num_envs = count
    manager.scene = SimpleNamespace(num_envs=count)
    manager.sim = SimpleNamespace(sim=SimpleNamespace(device="cpu"))
    manager.robot_list = [
        SimpleNamespace(robot_type="arm", arm_name=f"{side}_arm", gripper_name=f"{side}_ee",
                        ee_type="gripper", arm_joint_indices=[0, 1], gripper_joint_indices=[2, 3],
                        gripper_scale=[0, 1], gripper_move={"mimic": [0, -1, 0.5]})
        for side in ("left", "right")
    ]
    manager.robot_key = [Articulation(torch.arange(count * 4).reshape(count, 4).float() / 20 + offset)
                         for offset in (0.0, 0.1)]
    manager.process_name = lambda name: f"{name}_joint_state"
    manager.restore_name = lambda key: key.removesuffix("_joint_state")
    manager.get_robot_by_gripper_name = {
        robot.gripper_name: robot for robot in manager.robot_list
    }.__getitem__
    manager.get_robot_obs_name = lambda: [
        manager.process_name(name) for robot in manager.robot_list for name in (robot.arm_name, robot.gripper_name)
    ]
    manager.control_manager = ControlManager(count, manager)
    return manager


def run_queue_and_dispatch(manager, ids, actions):
    queue = manager.control_manager
    queue.push(ids, [[action] for action in actions])
    if manager.use_batched_cpu_controls:
        with manager.batch_control_gripper_reads(ids):
            controls = queue.pop(ids)
    else:
        controls = queue.pop(ids)
    manager.control_robot(controls)


class BatchCpuControlsTest(unittest.TestCase):
    def test_sparse_queue_and_targets_match_across_ticks(self):
        native, batch = make_manager(False), make_manager(True)
        ids = [3, 1]
        first = [
            {key: {"position": [value, 0.25]} for key in batch.get_robot_obs_name()}
            for value in (0.9, -0.1)
        ]
        second = [{key: value for key, value in action.items() if not key.startswith("left_")} for action in first]
        for actions in (first, second):
            for manager in (native, batch):
                run_queue_and_dispatch(manager, ids, actions)
                for articulation in manager.robot_key:
                    articulation.data.joint_pos[:, 2] += 0.1
            self.assertEqual(native.control_manager.prev_control, batch.control_manager.prev_control)
        for expected, actual in zip(native.robot_key, batch.robot_key, strict=True):
            self.assertEqual(len(expected.targets), len(actual.targets))
            for a, b in zip(expected.targets, actual.targets, strict=True):
                self.assertEqual(a[:2], b[:2])
                self.assertTrue(torch.equal(a[2], b[2]))
                self.assertTrue(torch.equal(a[3], b[3]))

    def test_snapshot_is_cleared_on_exception(self):
        manager = make_manager(True)
        with self.assertRaises(RuntimeError), manager.batch_control_gripper_reads([1]):
            manager.get_end_effector_real_val(manager.robot_list[0], [1])
            raise RuntimeError("queue failed")
        self.assertIsNone(manager._control_gripper_snapshot)

    def test_gpu_and_single_environment_keep_native_controls(self):
        single = make_manager(True, count=1)
        self.assertFalse(single.use_batched_cpu_controls)
        gpu = make_manager(True)
        gpu.sim.sim.device = torch.device("cuda:0")
        self.assertFalse(gpu.use_batched_cpu_controls)


if __name__ == "__main__":
    unittest.main()
