import ast
from contextlib import nullcontext
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np


class BatchedIKDispatchTests(unittest.TestCase):
    def test_two_arms_sharing_robot_name_keep_distinct_solutions(self):
        path = Path(__file__).parents[1] / "src/eval_client/eval_env.py"
        node = next(
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "take_action_batch"
        )
        ns = dict(os=os, np=np, deepcopy=deepcopy)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
        robots = [
            SimpleNamespace(
                type="target",
                robot_name="x5",
                arm_name=side + "_arm",
                gripper_name=side + "_ee",
                ee_type="gripper",
                gripper_move=dict(sign=1, mimic=[0, 1, 0]),
                gripper_scale=[0, 1],
            )
            for side in ("left", "right")
        ]
        calls = []
        controls = {}

        def solve(poses, indices, robot):
            calls.append((robot.arm_name, list(indices)))
            return [dict(status="Success", joint_value=np.full(6, pose[0])) for pose in poses]

        def process(info, index):
            controls[index] = info
            return [info]

        manager = SimpleNamespace(
            robot_list=robots,
            cpu_state=lambda **kw: nullcontext(),
            solve_ik_batch=solve,
            process_name=lambda name: name + "_joint_state",
            control_manager=SimpleNamespace(push=lambda *args: None),
        )
        env = SimpleNamespace(
            physx_monitor_enabled=False,
            num_envs=2,
            robot_manager=manager,
            take_action_cnt=[0, 0],
            step_lim=800,
            end_flag=[False, False],
            validate_action_dict=lambda action: None,
            get_action_type=lambda action: "ee",
            process_control_info=process,
            have_empty=lambda ids: True,
            reward_manager=SimpleNamespace(step=lambda **kw: None),
            is_episode_end=lambda: None,
        )
        actions = [
            dict(
                left_ee_pose=[i + 1] * 7,
                right_ee_pose=[i + 101] * 7,
                left_ee_joint_state=[0.5],
                right_ee_joint_state=[0.5],
            )
            for i in range(2)
        ]
        with patch.dict(os.environ, ROBODOJO_BATCH_IK="1"):
            ns["take_action_batch"](env, actions)
        self.assertEqual(calls, [("left_arm", [0, 1]), ("right_arm", [0, 1])])
        for i in range(2):
            np.testing.assert_array_equal(controls[i]["left_arm_joint_state"]["position"], np.full(6, i + 1))
            np.testing.assert_array_equal(controls[i]["right_arm_joint_state"]["position"], np.full(6, i + 101))
        self.assertEqual(env.take_action_cnt, [1, 1])


if __name__ == "__main__":
    unittest.main()
