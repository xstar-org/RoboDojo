"""CPU contracts for the actual EvalEnv reset and launcher readiness helper."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def eval_env_methods(events, client_class):
    source = ast.parse((ROOT / "src/eval_client/eval_env.py").read_text())
    factory = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "create_eval_env")
    original = next(n for n in factory.body if isinstance(n, ast.ClassDef))
    reset = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "reset")
    class Base:
        def reset(self, **kwargs):
            events.append("simulation_reset")
    cls = ast.ClassDef(name="TestEnv", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[reset], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    ns = dict(Base=Base, WsModelClient=client_class)
    exec(compile(module, str(ROOT / "src/eval_client/eval_env.py"), "exec"), ns)
    return ns["TestEnv"], original


class ReadyAfterScene(unittest.TestCase):
    def make_env(self, events, client_class):
        cls, original = eval_env_methods(events, client_class)
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

    def test_real_connection_after_complete_scene_reset(self):
        events = []
        class Client:
            def __init__(self, **kwargs):
                events.append("connect")
                self.kwargs = kwargs
            def call(self, **kwargs):
                events.append(("rpc", kwargs))
        env, _ = self.make_env(events, Client)
        env.reset([0])
        self.assertEqual(events, ["simulation_reset", "setup_scene", "origin", "robot_init", "reward_init", "connect", ("rpc", {"func_name": "reset"})])
        self.assertEqual(env.model_client.kwargs, env._model_client_kwargs)
        first_client = env.model_client
        env.reset([1])
        self.assertIs(env.model_client, first_client)
        self.assertEqual(events.count("connect"), 1)

    def test_connection_failure_leaves_safe_close_state(self):
        events = []
        def failed(**kwargs):
            raise ConnectionError("handshake failed")
        env, _ = self.make_env(events, failed)
        with self.assertRaisesRegex(ConnectionError, "handshake failed"):
            env.reset([0])
        self.assertIsNone(env.model_client)
        self.assertNotIn("rpc", events)
        main_source = ast.parse((ROOT / "src/eval_client/main.py").read_text())
        close = next(n for n in main_source.body if isinstance(n, ast.FunctionDef) and n.name == "_close_model_client")
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[close], type_ignores=[])), "actual_close", "exec"), ns)
        ns["_close_model_client"](env)

    def test_constructor_only_records_original_connection_parameters(self):
        events = []
        def forbidden(**kwargs):
            self.fail("constructor must not connect before scene initialization")
        _, original = self.make_env(events, forbidden)
        init = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "_model_client_kwargs" for t in n.targets))
        stop = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "model_client" for t in n.targets))
        block = ast.fix_missing_locations(ast.Module(body=init.body[start:stop + 1], type_ignores=[]))
        env = SimpleNamespace(deploy_cfg={"repeat_index": 4, "ws_ping_interval_s": 13, "ws_ping_timeout_s": 21})
        ns = dict(self=env, WsModelClient=forbidden, policy_server_url="ws://test", evaluation_id="eval", trial_id="trial", action_case_id="case")
        exec(compile(block, "actual_constructor", "exec"), ns)
        self.assertIsNone(env.model_client)
        self.assertEqual(env._model_client_kwargs, dict(url="ws://test", evaluation_id="eval", trial_id="trial", action_case_id="case", repeat_index=4, ws_ping_interval_s=13, ws_ping_timeout_s=21))

    def test_main_propagates_connect_failure_without_advancing_layout(self):
        main_source = ast.parse((ROOT / "src/eval_client/main.py").read_text())
        main = next(n for n in main_source.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        loop = next(n for n in main.body if isinstance(n, ast.While))
        actual_try = next(n for n in loop.body if isinstance(n, ast.Try))
        advanced = []
        def failed_reset(**kwargs):
            raise ConnectionError("native client retry exhausted")
        env = SimpleNamespace(env_seeds=[0], reset=failed_reset, seed_manager=SimpleNamespace(eval_step=lambda: advanced.append(True)))
        ns = dict(env=env, enable_monitor=False,
                  PhysXFatalError=type("PhysXFatalError", (Exception,), {}),
                  PhysXBrokenError=type("PhysXBrokenError", (Exception,), {}),
                  UnStableError=type("UnStableError", (Exception,), {}))
        module = ast.fix_missing_locations(ast.Module(body=[actual_try], type_ignores=[]))
        with self.assertRaisesRegex(ConnectionError, "native client retry exhausted"):
            exec(compile(module, str(ROOT / "src/eval_client/main.py"), "exec"), ns)
        self.assertEqual(advanced, [])


if __name__ == "__main__":
    unittest.main()
