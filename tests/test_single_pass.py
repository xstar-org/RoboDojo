"""Tests for single-pass capture and render_passes config."""
import ast
import os
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "env/camera_manager/capture/tiled_capture_manager.py"


class SinglePassTests(unittest.TestCase):
    def test_render_passes_env_var(self):
        text = SRC.read_text(encoding="utf-8")
        self.assertIn("ROBODOJO_CAMERA_RENDER_PASSES", text)
        self.assertIn('int(os.environ.get', text)
        self.assertIn('if self.render_passes not in (1, 2)', text)

    def test_step_uses_render_passes(self):
        text = SRC.read_text(encoding="utf-8")
        self.assertIn("min_passes=self.render_passes", text,
                      "step() must use configurable render_passes, not hardcoded 2")

    def test_products_gated(self):
        """Products must be disabled by default and only enabled during capture."""
        text = SRC.read_text(encoding="utf-8")
        self.assertIn("set_updates_enabled(False)", text,
                      "products must start disabled")
        self.assertIn("set_updates_enabled(True)", text,
                      "products must be enabled during capture")
        # The finally-style cleanup after the loop
        self.assertIn("if last_view is not None:", text,
                      "last product must be disabled after capture loop")

    def test_view_mapping(self):
        """_view_of_cam must exist and be used in _read_camera."""
        text = SRC.read_text(encoding="utf-8")
        self.assertIn("_view_of_cam", text)
        self.assertIn("view_id = self._view_of_cam[cam_id]", text)

    def test_no_retargeting(self):
        """The old retargeting (SetTargets on shared product) must be gone."""
        text = SRC.read_text(encoding="utf-8")
        self.assertNotIn("SetTargets", text,
                         "retargeting must be removed; use independent products")

    def test_physics_guard(self):
        text = SRC.read_text(encoding="utf-8")
        self.assertIn("Physics advanced during sequential camera capture", text)


if __name__ == "__main__":
    unittest.main()
