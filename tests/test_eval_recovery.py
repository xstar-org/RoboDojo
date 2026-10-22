import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.camera_observation import (
    InvalidCameraObservationError, capture_valid_observation, invalid_color_buffers,
)
from utils.eval_lifecycle import run_evaluation
from utils.save_file import VideoStreamWriter, abort_active_video_streams


def observation(value=0):
    return {0: {'vision': {'cam_head': {'color': np.full((2, 2, 3), value, np.uint8)}}}}


class CameraRecoveryTests(unittest.TestCase):
    def test_valid_frame_returns_without_rendering_or_copying(self):
        obs = observation(100)
        render = Mock()
        self.assertIs(capture_valid_observation(lambda: obs, render), obs)
        render.assert_not_called()

    def test_black_frame_is_refreshed_before_returning(self):
        good = observation(30)
        capture = Mock(side_effect=[observation(), good])
        render = Mock()
        self.assertIs(capture_valid_observation(capture, render), good)
        self.assertEqual(render.call_count, 4)

    def test_persistent_black_frame_raises_after_bounded_refresh(self):
        render = Mock()
        with self.assertRaisesRegex(InvalidCameraObservationError, '64 render-only'):
            capture_valid_observation(lambda: observation(), render)
        self.assertEqual(render.call_count, 64)

    def test_empty_buffer_is_invalid_and_other_camera_is_checked(self):
        obs = observation(100)
        obs[0]['vision']['cam_left_wrist'] = {'color': np.empty((0, 0, 3), np.uint8)}
        self.assertEqual(invalid_color_buffers(obs), [(0, 'cam_left_wrist')])

    def test_nonzero_dark_float_camera_is_not_classified_as_black(self):
        obs = observation()
        obs[0]['vision']['cam_head']['color'] = np.full((2, 2, 3), .001)
        self.assertEqual(invalid_color_buffers(obs), [])

    def test_state_only_observation_is_unchanged(self):
        obs = {0: {'state': {'joint': [0.]}}}
        self.assertIs(capture_valid_observation(lambda: obs, Mock()), obs)


class VideoAbortTests(unittest.TestCase):
    def test_abort_kills_encoder_before_closing_buffered_stdin(self):
        order = []
        process = Mock()
        process.kill.side_effect = lambda: order.append('kill')
        process.wait.side_effect = lambda **kw: order.append('wait')
        process.stdin.close.side_effect = lambda: order.append('close')
        with tempfile.TemporaryDirectory() as directory, patch('utils.save_file.subprocess.Popen', return_value=process):
            writer = VideoStreamWriter(str(Path(directory) / 'video.mp4'), 2, 2, 3)
            abort_active_video_streams()
            self.assertEqual(order, ['kill', 'wait', 'close'])
            process.wait.assert_called_once_with(timeout=5)
            self.assertIsNone(writer.proc)
            writer.abort()  # Repeated cleanup is safe.
            self.assertEqual(order, ['kill', 'wait', 'close'])


class LifecycleTests(unittest.TestCase):
    def run_child(self, source, *args):
        return subprocess.run(
            [sys.executable, '-c', source, *args], cwd=ROOT,
            capture_output=True, text=True, timeout=5,
        )

    def test_successful_main_returns_normally(self):
        main, cleanup = Mock(), Mock()
        run_evaluation(main, cleanup)
        main.assert_called_once_with()
        cleanup.assert_not_called()

    def test_exception_exits_without_running_failing_native_atexit(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = str(Path(directory) / 'native-atexit')
            aborted = str(Path(directory) / 'aborted')
            r = self.run_child('''
import atexit, pathlib, sys
from utils.eval_lifecycle import run_evaluation
atexit.register(lambda: pathlib.Path(sys.argv[1]).write_text('unexpected'))
def main(): raise BrokenPipeError('encoder exited')
def cleanup(): pathlib.Path(sys.argv[2]).write_text('aborted')
run_evaluation(main, cleanup, cleanup_timeout=.5)
''', marker, aborted)
            self.assertEqual(r.returncode, 1, r.stderr)
            self.assertIn('BrokenPipeError: encoder exited', r.stderr)
            self.assertFalse(Path(marker).exists())
            self.assertEqual(Path(aborted).read_text(), 'aborted')

    def test_system_exit_retry_code_is_preserved(self):
        r = self.run_child('''
from utils.eval_lifecycle import run_evaluation
def main(): raise SystemExit(99)
run_evaluation(main, lambda: None)
''')
        self.assertEqual(r.returncode, 99, r.stderr)

    def test_hung_native_cleanup_is_bounded(self):
        r = self.run_child('''
import time
from utils.eval_lifecycle import bounded_cleanup
with bounded_cleanup(.1, exit_code=7): time.sleep(30)
''')
        self.assertEqual(r.returncode, 7, r.stderr)
        self.assertIn('cleanup timed out', r.stderr)

    def test_hung_exception_cleanup_is_also_bounded(self):
        r = self.run_child('''
import time
from utils.eval_lifecycle import run_evaluation
def main(): raise RuntimeError('failed')
run_evaluation(main, lambda: time.sleep(30), cleanup_timeout=.1)
''')
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn('cleanup timed out', r.stderr)


if __name__ == '__main__':
    unittest.main()
