import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "compare_replay", Path(__file__).resolve().parents[1] / "scripts/compare_replay_throughput.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class ThroughputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_data(self, name, envs, wall, spp=64, version="5.1.0"):
        path = self.root / name
        path.mkdir()
        settings = {"/rtx/pathtracing/spp": spp}
        (path / "FINISHED").touch()
        (path / "runtime-profile.json").write_text(json.dumps({
            "settings": settings, "versions": {"isaacsim": version},
        }))
        (path / "benchmark.json").write_text(json.dumps({
            "steps": 96, "num_envs": envs, "episode_wall_s": wall,
            "step_total_s": {"mean": wall / 96},
            "tape_sha256": "same-actions", "render_profile": settings,
        }))
        return path

    def test_environment_normalization_and_minimum_gain(self):
        baseline = self.run_data("one", 1, 240)
        slow = self.run_data("four-slow", 4, 1000)
        fast = self.run_data("four-fast", 4, 480)
        self.assertFalse(gate.compare(baseline, slow, 1.5)["passed"])
        self.assertEqual(gate.compare(baseline, fast, 1.5)["speedup"], 2)

    def test_lower_spp_or_other_sim_version_is_not_a_speedup(self):
        baseline = self.run_data("one", 1, 240)
        for name, options in (("low-spp", {"spp": 16}), ("sim6", {"version": "6.0.0"})):
            with self.subTest(name=name), self.assertRaises(ValueError):
                gate.compare(baseline, self.run_data(name, 4, 100, **options), 1.5)

    def test_partial_or_failed_run_cannot_pass(self):
        baseline = self.run_data("one", 1, 240)
        candidate = self.run_data("four", 4, 100)
        (candidate / "FAILED.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "failed replay"):
            gate.compare(baseline, candidate, 1.5)


if __name__ == "__main__":
    unittest.main()
