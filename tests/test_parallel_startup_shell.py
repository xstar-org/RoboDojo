"""CPU integration checks for the real shell orchestrator (requires bash)."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'requires POSIX bash')
class ParallelStartup(unittest.TestCase):
    def run_case(self, fail_server):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / 'XPolicyLab/policy/Test'
            policy.mkdir(parents=True)
            utils = root / 'XPolicyLab/utils'
            utils.mkdir()
            events = root / 'events'
            events.mkdir()
            scripts = {
                policy / 'setup_eval_policy_server.sh': '''sleep 0.15
if [ "$FAIL_SERVER" = 1 ]; then exit 7; fi
[ -f "$EVENTS/scene_initialized" ] || exit 8
touch "$EVENTS/model_loaded"
sleep 30
''',
                policy / 'setup_eval_env_client.sh': '''echo $$ > "$EVENTS/client_pid"
touch "$EVENTS/scene_initialized"
while [ ! -f "$ROBODOJO_POLICY_READY_FILE" ]; do sleep 0.01; done
[ -f "$EVENTS/model_loaded" ] || exit 9
touch "$EVENTS/connected"
''',
                utils / 'get_free_port.sh': 'echo 12345\n',
                utils / 'wait_for_policy_server.sh': '''while [ ! -f "$EVENTS/model_loaded" ]; do
kill -0 "$3" 2>/dev/null || exit 1
sleep 0.01
done
''',
            }
            for path, content in scripts.items():
                path.write_text('#!/usr/bin/env bash\nset -eu\n' + content)
            orchestrator = Path(__file__).resolve().parents[1] / 'scripts/internal/run_policy_eval.sh'
            result = subprocess.run(['bash', str(orchestrator), str(policy), 'bench', 'task', 'checkpoint',
                                     'config', 'action', '1', '1', '0', 'policy-env', 'sim-env'],
                                    env=dict(os.environ, EVENTS=str(events), FAIL_SERVER=str(int(fail_server))),
                                    capture_output=True, text=True, timeout=8)
            self.assertTrue((events / 'scene_initialized').exists(), result.stdout + result.stderr)
            self.assertEqual((events / 'connected').exists(), not fail_server, result.stdout + result.stderr)
            self.assertEqual(result.returncode == 0, not fail_server, result.stdout + result.stderr)
            pid = int((events / 'client_pid').read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_simulation_precedes_model_ready(self):
        self.run_case(False)

    def test_server_failure_cleans_waiting_client(self):
        self.run_case(True)


if __name__ == '__main__':
    unittest.main()
