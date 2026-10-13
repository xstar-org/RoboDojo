"""Completion checks must preserve scores without reading simulation state."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest


class RewardSnapshotBoundaryTests(unittest.TestCase):
    def make_manager(self):
        path = Path(__file__).parents[1] / "env/reward_manager/reward_manager.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardManager")
        names = {"get_reward", "_snapshot_get_reward", "get_score", "_score_for_env", "_score_for_state"}
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        ns = {"List": list}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
        manager = ns["RewardManager"]()
        events = []

        @contextmanager
        def snapshot(**kwargs):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        manager.env = SimpleNamespace(
            scene_manager=SimpleNamespace(layout_manager=SimpleNamespace(cpu_poses=snapshot)),
            robot_manager=SimpleNamespace(cpu_state=snapshot),
        )
        manager.num_envs = 2
        manager.func_parser = SimpleNamespace(_check_env_success=lambda i: True)
        manager.check_list = [[], ["pending"]]
        for name in ("final_check_list", "trigger_check_list", "query_list", "trigger_query_list"):
            setattr(manager, name, [[], []])
        manager.score_meta = [{"mode": "paired", "gradient": [50, 100]} for _ in range(2)]
        manager.score_achieved = [[50, 50], [50, 50]]
        manager.score_completed_count = [2, 2]
        manager.final_score_meta = [{"mode": None} for _ in range(2)]
        manager.final_score_achieved = [[], []]
        manager.final_score_completed_count = [0, 0]
        manager._final_check = lambda: events.append("final_check")
        manager._final_score = lambda: events.append("final_score")
        return manager, events

    def test_completion_and_scores_without_physics_reads(self):
        manager, events = self.make_manager()
        self.assertEqual(manager.get_reward(False), [1.0, 0.0])
        self.assertEqual(manager._gated_score_lst, [100.0, 50.0])
        self.assertEqual(events, [])

    def test_final_predicates_still_run_inside_snapshots(self):
        manager, events = self.make_manager()
        self.assertEqual(manager.get_reward(True), [1.0, 0.0])
        self.assertEqual(events, ["enter", "enter", "final_check", "final_score", "exit", "exit"])

    def test_failed_environment_remains_failed(self):
        manager, events = self.make_manager()
        manager.func_parser._check_env_success = lambda i: i != 0
        self.assertEqual(manager.get_reward(False), [0.0, 0.0])
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
