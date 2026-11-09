"""Exercise evaluation settling without initializing Isaac Sim."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


def setup_scene_method():
    path = Path(__file__).resolve().parents[1] / "src/eval_client/eval_env.py"
    tree = ast.parse(path.read_text())
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "setup_scene"]
    assert len(methods) == 1
    scope = {"UnStableError": RuntimeError}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
    return scope["setup_scene"]


class WarmupObservationTests(unittest.TestCase):
    def environment(self, config):
        return SimpleNamespace(
            num_envs=2,
            sim_config=config,
            unstable_nums=0,
            episode_nums=2,
            physx_monitor_enabled=False,
            scene_manager=SimpleNamespace(
                apply_saved_poses=Mock(),
                layout_manager=SimpleNamespace(check_layout_stability=Mock(return_value=(True, []))),
            ),
            _align_layout_success=Mock(),
            render=Mock(),
            sim_step=Mock(),
            obs_manager=SimpleNamespace(render_for_capture=Mock(), get_obs=Mock()),
        )

    def test_default_and_explicit_enable_preserve_native_capture_schedule(self):
        for config in ({}, {"capture_eval_warmup_observations": True}):
            with self.subTest(config=config):
                env = self.environment(config)
                setup_scene_method()(env)
                self.assertEqual(env.sim_step.call_count, 200)
                self.assertEqual(env.render.call_count, 10)
                self.assertEqual(env.obs_manager.render_for_capture.call_count, 40)
                self.assertEqual(env.obs_manager.get_obs.call_count, 40)

    def test_disable_skips_captures_and_preserves_settling_and_stability_check(self):
        env = self.environment({"capture_eval_warmup_observations": False})
        setup_scene_method()(env)
        env.scene_manager.apply_saved_poses.assert_called_once_with(env_idx_list=[0, 1])
        env.scene_manager.layout_manager.check_layout_stability.assert_called_once_with(env)
        self.assertEqual(env.sim_step.call_count, 200)
        self.assertEqual(env.render.call_count, 10)
        env.obs_manager.render_for_capture.assert_not_called()
        env.obs_manager.get_obs.assert_not_called()

    def test_disable_does_not_bypass_native_unstable_failure(self):
        env = self.environment({"capture_eval_warmup_observations": False})
        env.scene_manager.layout_manager.check_layout_stability.return_value = (False, [0, 1])
        with self.assertRaisesRegex(RuntimeError, "All scene Unstable"):
            setup_scene_method()(env)
        self.assertEqual(env.unstable_nums, 2)
        self.assertEqual(env.episode_nums, 0)
        env.sim_step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
