"""Tests for the render-action pipeline."""
import ast
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TCM = REPO / "env/camera_manager/capture/tiled_capture_manager.py"
ENV = REPO / "src/eval_client/eval_env.py"


class RenderPipelineTests(unittest.TestCase):
    def test_start_capture_exists(self):
        text = TCM.read_text(encoding="utf-8")
        self.assertIn("def start_capture", text)
        self.assertIn("def finish_capture", text)

    def test_start_enables_products(self):
        text = TCM.read_text(encoding="utf-8")
        fn = text[text.find("def start_capture"):text.find("def finish_capture")]
        self.assertIn("set_updates_enabled(True)", fn)

    def test_finish_waits_and_reads(self):
        text = TCM.read_text(encoding="utf-8")
        fn = text[text.find("def finish_capture"):text.find("def _disable_pipeline")]
        self.assertIn("wait_for_latest_cameras", fn)
        self.assertIn("_read_camera", fn)
        self.assertIn("set_updates_enabled(False)", fn)

    def test_eval_env_calls_start_after_physics(self):
        text = ENV.read_text(encoding="utf-8")
        # start_capture should appear between the physics step loop and reward
        reward_pos = text.find("self.reward_manager.step")
        start_pos = text.find("self.obs_manager.capture_manager.start_capture()")
        self.assertGreater(reward_pos, 0)
        self.assertGreater(start_pos, 0)
        self.assertLess(start_pos, reward_pos,
                        "start_capture must be called before reward_manager.step")

    def test_eval_env_uses_finish_in_get_obs(self):
        text = ENV.read_text(encoding="utf-8")
        self.assertIn("finish_capture", text,
                      "get_obs_batch should use finish_capture for pipelined reads")

    def test_physics_guard_preserved(self):
        text = TCM.read_text(encoding="utf-8")
        self.assertIn("Physics advanced during pipelined camera capture", text)

    def test_fallback_to_sequential(self):
        """If start_capture wasn't called, finish_capture falls back to step()."""
        text = TCM.read_text(encoding="utf-8")
        fn = text[text.find("def finish_capture"):text.find("def _disable_pipeline")]
        self.assertIn("self.step(", fn,
                      "finish_capture must fall back to step() when pipeline not started")


if __name__ == "__main__":
    unittest.main()
