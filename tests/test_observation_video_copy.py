import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class ObservationVideoCopyTests(unittest.TestCase):
    def test_rgb_video_and_policy_share_owned_contiguous_pixels(self):
        source = Path(__file__).parents[1] / "src/eval_client/eval_env.py"
        tree = ast.parse(source.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_obs_batch")
        namespace = {"deepcopy": deepcopy}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        rgba = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
        rgb = rgba[:, :, :3]
        original = {i: {"vision": {"head": {"color": rgb}}, "state": {"x": [i]}} for i in range(2)}
        streamed = []
        env = SimpleNamespace(
            num_envs=2, physx_monitor_enabled=False, end_flag=[False, True],
            obs_manager=SimpleNamespace(render_for_capture=lambda: None, get_obs=lambda **kwargs: original),
            _stream_vision=lambda i, data: streamed.append((i, data)),
        )
        result = namespace["get_obs_batch"](env)
        self.assertEqual([i for i, _ in streamed], [0])
        image = result[0]["vision"]["head"]["color"]
        np.testing.assert_array_equal(image, rgb)
        self.assertTrue(image.flags.c_contiguous)
        self.assertIs(streamed[0][1]["vision"]["head"]["color"], image)
        self.assertFalse(np.shares_memory(image, rgba))
        rgba[:] = 0
        self.assertGreater(image.sum(), 0)
        result[0]["state"]["x"][0] = 9
        self.assertEqual(original[0]["state"]["x"], [0])
        streamed.clear()
        namespace["get_obs_batch"](env, env_idx_list=[1], last_frame=True)
        self.assertEqual([i for i, _ in streamed], [1])


if __name__ == "__main__":
    unittest.main()
