"""CPU integration checks for the real shell orchestrator (POSIX Bash5.1+)."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires POSIX bash")
class ParallelStartup(unittest.TestCase):
    def run_case(self, mode, expected):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "XPolicyLab/policy/Test"
            policy.mkdir(parents=True)
            utils = root / "XPolicyLab/utils"
            utils.mkdir()
            events = root / "events"
            events.mkdir()
            scripts = {
                policy / "setup_eval_policy_server.sh": '''echo $$ > "$EVENTS/server_pid"
sleep 0.15
if [ "$MODE" = server_failure ]; then exit 7; fi
[ -f "$EVENTS/scene_initialized" ] || exit 8
touch "$EVENTS/model_loaded"
if [[ "$MODE" == both_* || "$MODE" == already_* || "$MODE" == force_no_job_* ]]; then
  while [ ! -f "$EVENTS/client_finishing" ]; do sleep 0.01; done
  exit 0
fi
sleep 30
''',
                policy / "setup_eval_env_client.sh": '''echo $$ > "$EVENTS/client_pid"
[ -z "${ROBODOJO_POLICY_READY_FILE+x}" ] || exit 90
touch "$EVENTS/scene_initialized"
if [ "$MODE" = client_failure_before_connect ]; then sleep 0.03; exit 23; fi
# Mimic the existing WsModelClient retries, not a launcher ready-file protocol.
while [ ! -f "$EVENTS/model_loaded" ]; do sleep 0.01; done
touch "$EVENTS/connected"
if [ "$MODE" = client_failure_after_connect ]; then exit 24; fi
if [[ "$MODE" == both_* || "$MODE" == already_* || "$MODE" == force_no_job_* ]]; then touch "$EVENTS/client_finishing"; fi
if [[ "$MODE" == *_client_failure ]]; then exit 29; fi
''',
                utils / "get_free_port.sh": "echo 12345\n",
                utils / "wait_for_policy_server.sh": "exit 91\n",
            }
            for path, content in scripts.items():
                path.write_text("#!/usr/bin/env bash\nset -eu\n" + content)
            orchestrator = Path(__file__).resolve().parents[1] / "scripts/internal/run_policy_eval.sh"
            env = dict(os.environ, EVENTS=str(events), MODE=mode)
            env.pop("ROBODOJO_POLICY_READY_FILE", None)
            if mode.startswith(("already_", "force_no_job_")):
                bash_env = root / "delay_wait.sh"
                bash_env.write_text('''wait() {
  if [[ "${1:-}" = -n ]]; then
    sleep 0.5
    for child in "${@:4}"; do
      if kill -0 "$child" 2>/dev/null; then return 92; fi
    done
    touch "$EVENTS/children_exited_before_wait"
    if [[ "$MODE" == force_no_job_* ]]; then
      unset "$3"
      touch "$EVENTS/forced_no_wait_job"
      return 127
    fi
  fi
  builtin wait "$@"
}
''')
                env["BASH_ENV"] = str(bash_env)
            result = subprocess.run(["bash", str(orchestrator), str(policy), "bench", "task", "checkpoint",
                                     "config", "action", "1", "1", "0", "policy-env", "sim-env"],
                                    env=env, capture_output=True, text=True, timeout=8)
            self.assertTrue((events / "scene_initialized").exists(), result.stdout + result.stderr)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            self.assertEqual((events / "connected").exists(), mode not in ("server_failure", "client_failure_before_connect"))
            if mode.startswith(("already_", "force_no_job_")):
                self.assertTrue((events / "children_exited_before_wait").exists(), result.stdout + result.stderr)
            if mode.startswith("force_no_job_"):
                self.assertTrue((events / "forced_no_wait_job").exists(), result.stdout + result.stderr)
            for name in ("client_pid", "server_pid"):
                pid = int((events / name).read_text())
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_simulation_precedes_model_listen(self):
        self.run_case("success", 0)

    def test_server_failure_cleans_waiting_client(self):
        self.run_case("server_failure", 1)

    def test_client_failure_before_connection_preserves_status(self):
        self.run_case("client_failure_before_connect", 23)

    def test_client_failure_after_connection_preserves_status(self):
        self.run_case("client_failure_after_connect", 24)

    def test_near_simultaneous_exits_preserve_client_failure(self):
        self.run_case("both_client_failure", 29)

    def test_near_simultaneous_success_is_client_completion(self):
        self.run_case("both_success", 0)

    def test_both_children_already_exited_before_wait_success(self):
        self.run_case("already_success", 0)

    def test_both_children_already_exited_before_wait_failure(self):
        self.run_case("already_client_failure", 29)

    def test_explicit_no_job_unset_branch_success(self):
        self.run_case("force_no_job_success", 0)

    def test_explicit_no_job_unset_branch_client_failure(self):
        self.run_case("force_no_job_client_failure", 29)


if __name__ == "__main__":
    unittest.main()
