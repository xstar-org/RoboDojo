"""Delivery failures must never silently hand stale RGB to the policy."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "render_sync", Path(__file__).resolve().parents[1] / "env/camera_manager/capture/render_sync.py"
)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class DeliveryTests(unittest.TestCase):
    def test_stalled_dispatcher_raises_without_reading_old_buffers(self):
        env = SimpleNamespace(render=Mock())
        with patch.object(sync, "zero_delay_active", return_value=True), \
             patch.object(sync, "dispatcher_time", return_value=42):
            with self.assertRaisesRegex(RuntimeError, "received 0 of 2"):
                sync.wait_for_latest_cameras(env, min_passes=2, max_passes=3)
        self.assertEqual(env.render.call_count, 3)

    def test_one_current_delivery_and_two_frame_legacy_pipeline(self):
        for zero_delay, expected in ((True, 1), (False, 2)):
            env = SimpleNamespace(render=Mock())
            with self.subTest(zero_delay=zero_delay), \
                 patch.object(sync, "zero_delay_active", return_value=zero_delay), \
                 patch.object(sync, "dispatcher_time", side_effect=range(20)):
                self.assertEqual(sync.wait_for_latest_cameras(env), expected)
            self.assertEqual(env.render.call_count, expected)


if __name__ == "__main__":
    unittest.main()
