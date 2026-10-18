"""CPU contracts for the actual EvalEnv reset and launcher readiness helper."""
import ast
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("policy_ready", ROOT / "src/eval_client/policy_ready.py")
ready_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ready_module)


def eval_env_methods(events, client_class, wait):
    source = ast.parse((ROOT / "src/eval_client/eval_env.py").read_text())
    factory = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "create_eval_env")
    original = next(n for n in factory.body if isinstance(n, ast.ClassDef))
    reset = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "reset")
    class Base:
        def reset(self, **kwargs):
            events.append("simulation_reset")
    cls = ast.ClassDef(name="TestEnv", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[reset], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    ns = dict(Base=Base, WsModelClient=client_class, wait_for_policy_ready=wait)
    exec(compile(module, str(ROOT / "src/eval_client/eval_env.py"), "exec"), ns)
    return ns["TestEnv"], original


class ReadyAfterScene(unittest.TestCase):
    def make_env(self, events, client_class, wait):
        cls, original = eval_env_methods(events, client_class, wait)
        env = cls()
        env.num_envs = 1
        env.model_client = None
        env._model_client_kwargs = dict(url="ws://test", trial_id="trial")
        env._abort_video_writers = lambda: None
        env.scene_manager = SimpleNamespace(layout_manager=SimpleNamespace(set_saved_layout=lambda *a: None))
        env.seed_manager = SimpleNamespace(get_seed_scene_info=lambda seed: seed)
        env.obs_manager = SimpleNamespace(reset=lambda: None)
        env.setup_scene = lambda: events.append("setup_scene")
        env.robot_manager = SimpleNamespace(set_origin_endpose=lambda: events.append("origin"), set_robot_init_state=lambda: events.append("robot_init"))
        env.reward_manager = SimpleNamespace(init_state=lambda: events.append("reward_init"))
        return env, original

    def test_wait_then_real_connection_after_complete_scene_reset(self):
        events = []
        class Client:
            def __init__(self, **kwargs):
                events.append("connect")
                self.kwargs = kwargs
            def call(self, **kwargs):
                events.append(("rpc", kwargs))
        env, _ = self.make_env(events, Client, lambda: events.append("wait_ready"))
        env.reset([0])
        self.assertEqual(events, ["simulation_reset", "setup_scene", "origin", "robot_init", "reward_init", "wait_ready", "connect", ("rpc", {"func_name": "reset"})])
        self.assertEqual(env.model_client.kwargs, env._model_client_kwargs)
        first_client = env.model_client
        env.reset([1])
        self.assertIs(env.model_client, first_client)
        self.assertEqual(events.count("connect"), 1)
        self.assertEqual(events.count("wait_ready"), 1)

    def test_timeout_prevents_connection_and_rpc(self):
        events = []
        def forbidden(**kwargs):
            self.fail("client must not connect before READY")
        def timeout():
            raise TimeoutError("model not ready")
        env, _ = self.make_env(events, forbidden, timeout)
        with self.assertRaises(TimeoutError):
            env.reset([0])
        self.assertEqual(events[-1], "reward_init")
        self.assertIsNone(env.model_client)

    def test_connection_failure_leaves_safe_close_state(self):
        events = []
        def failed(**kwargs):
            raise ConnectionError("handshake failed")
        env, _ = self.make_env(events, failed, lambda: events.append("wait_ready"))
        with self.assertRaisesRegex(ConnectionError, "handshake failed"):
            env.reset([0])
        self.assertIsNone(env.model_client)
        self.assertNotIn("rpc", events)
        main_source = ast.parse((ROOT / "src/eval_client/main.py").read_text())
        close = next(n for n in main_source.body if isinstance(n, ast.FunctionDef) and n.name == "_close_model_client")
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[close], type_ignores=[])), "actual_close", "exec"), ns)
        ns["_close_model_client"](env)

    def test_constructor_preserves_standalone_eager_connection(self):
        events = []
        class Client:
            def __init__(self, **kwargs):
                events.append(kwargs)
        _, original = self.make_env(events, Client, lambda: None)
        init = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "model_client" for t in n.targets))
        block = ast.fix_missing_locations(ast.Module(body=init.body[start:start + 2], type_ignores=[]))
        env = SimpleNamespace(_model_client_kwargs={"url": "ws://test"})
        ns = dict(self=env, os=os, WsModelClient=Client)
        with patch.dict(os.environ, {}, clear=True):
            exec(compile(block, "actual_constructor", "exec"), ns)
        self.assertIsInstance(env.model_client, Client)
        with patch.dict(os.environ, {"ROBODOJO_POLICY_READY_FILE": "pending"}, clear=True):
            exec(compile(block, "actual_constructor", "exec"), ns)
        self.assertIsNone(env.model_client)
        self.assertEqual(len(events), 1)


class ReadyMarker(unittest.TestCase):
    def test_direct_client_has_no_wait(self):
        with patch.dict(os.environ, {}, clear=True):
            ready_module.wait_for_policy_ready()

    def test_only_marker_releases_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "SERVER_READY"
            with patch.dict(os.environ, {"ROBODOJO_POLICY_READY_FILE": str(marker), "ROBODOJO_POLICY_READY_TIMEOUT": "0"}, clear=True):
                with self.assertRaises(TimeoutError):
                    ready_module.wait_for_policy_ready()
                marker.touch()
                ready_module.wait_for_policy_ready()


if __name__ == "__main__":
    unittest.main()
