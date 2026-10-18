"""Start simulation while the policy loads, connecting only after launcher READY."""
from pathlib import Path
import time


def install_client_gate(module, ready_path, timeout_s=1200, poll_s=0.1):
    original = module.WsModelClient
    if getattr(original, '_codex_launcher_ready_gate', False):
        raise RuntimeError('Launcher READY gate already installed')
    ready_path = Path(ready_path)

    class LauncherReadyClient:
        _codex_launcher_ready_gate = True

        def __init__(self, **kwargs):
            self._kwargs = dict(kwargs)
            self._client = None
            self._closed = False

        def _connected(self):
            if self._closed:
                raise RuntimeError('Policy client is closed')
            if self._client is None:
                deadline = time.monotonic() + timeout_s
                while not ready_path.is_file():
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Policy model READY timeout: ' + str(ready_path))
                    time.sleep(poll_s)
                self._client = original(**self._kwargs)
            return self._client

        def call(self, *args, **kwargs):
            return self._connected().call(*args, **kwargs)

        def __getattr__(self, name):
            if name.startswith('_'):
                raise AttributeError(name)
            if name in ('trial_id', 'action_case_id', 'repeat_index') and self._client is None:
                return self._kwargs.get(name)
            return getattr(self._connected(), name)

        def close(self):
            if not self._closed:
                self._closed = True
                if self._client is not None:
                    self._client.close()

    module.WsModelClient = LauncherReadyClient
    return LauncherReadyClient


def main():
    import os
    import sys
    import client_server.ws.model_client as client_module

    ready = os.environ["ROBODOJO_POLICY_READY_FILE"]
    install_client_gate(client_module, ready,
                        timeout_s=float(os.environ.get("ROBODOJO_POLICY_READY_TIMEOUT", "600")))
    target = Path(__file__).resolve().parents[2] / "src/eval_client/main.py"
    # Keep this launcher as argv[0]: main.py's fatal execv restart must reinstall the gate.
    sys.path.insert(0, str(target.parents[2]))
    exec(compile(target.read_bytes(), str(target), "exec"),
         {"__name__": "__main__", "__file__": str(target), "__package__": None})


if __name__ == "__main__":
    main()
