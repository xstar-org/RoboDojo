"""CPU regressions for scheduling, native outcomes and runtime adapters."""

import ast
import asyncio
from collections import OrderedDict
import contextvars
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runtime")]
from balanced_model_plan import balanced_chunks, balanced_slots, next_assignment
import client_runtime
from gripper_snapshot import install
from multigpu_plan import MetricSteps
from native_accounting import collect_attempt
from policy_compat import repair_prompt_cache


class SchedulingTests(unittest.TestCase):
    def test_nine_workers_own_exactly_fifty_cases_and_resume_remaining(self):
        plans = {"tic": {"seeds": [{"seed": 1, "layout_ids": list(range(50))}]}}
        self.assertEqual([len(ids) for _, ids in balanced_chunks(plans["tic"], 6)], [6] * 5 + [5] * 4)
        slots = balanced_slots({"1": [1], "2": [2], "3": [3]}, [1, 2, 3], 3)
        self.assertEqual(len(slots), 9)
        active = []
        confirmed = {"tic": {(1, 0), (1, 13), (1, 49)}}
        while assignment := next_assignment(plans, confirmed, active, ["tic"], {"tic": 6}):
            task, seed, ids = assignment
            active.append(dict(task=task, seed=seed, layout_ids=ids))
        owned = [(q["seed"], i) for q in active for i in q["layout_ids"]]
        self.assertEqual(len(owned), len(set(owned)))
        self.assertEqual(set(owned) | confirmed["tic"], {(1, i) for i in range(50)})
        self.assertFalse(set(owned) & confirmed["tic"])

    def test_counter_retains_incomplete_lines_and_handles_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "metrics.jsonl"
            counter = MetricSteps(p)
            p.write_text(json.dumps(dict(kind="take_action_batch", counts=[32, 16])) + "\n" + '{"kind":')
            self.assertEqual(counter.poll(), 48)
            with p.open("a") as f:
                f.write('"reset"}\n' + json.dumps(dict(kind="take_action_batch", counts=[5, 7])) + "\n")
            self.assertEqual(counter.poll(), 12)
            self.assertEqual(counter.total, 60)

    def test_missing_native_reply_is_pending_not_a_policy_failure(self):
        rows, excluded = {}, {}
        data = {"details": {"0": {"layout_id": 0, "success": True, "score": 1}}, "eval_time": 1}
        self.assertFalse(
            collect_attempt(rows, excluded, worker=0, seed=1, requested=[0, 1], data=data, log="", attempt=0)
        )
        self.assertEqual(set(rows), {(1, 0)})
        self.assertEqual(excluded, {})
        self.assertTrue(
            collect_attempt(
                rows,
                excluded,
                worker=0,
                seed=1,
                requested=[0, 1],
                data=data,
                log="Success nums: 1, Fail nums: 0, Unstable nums: 1",
                attempt=0,
            )
        )
        self.assertEqual(set(excluded), {(1, 1)})


class RuntimeTests(unittest.TestCase):
    def test_layout_selection_is_per_environment_and_respects_resume_filter(self):
        def make():
            return SimpleNamespace(
                seed_manager=SimpleNamespace(
                    seed_info=dict.fromkeys(range(5)), seed_list=[0, 2, 4], st_idx=0, ed_idx=3, idx=0
                )
            )

        with (
            patch.dict(os.environ, OW_LAYOUT_IDS="[4, 1, 2]"),
            patch.object(client_runtime, "_record_metrics"),
            patch.object(client_runtime, "_verify_runtime"),
            patch.object(client_runtime, "_install_optimizations"),
        ):
            env = client_runtime.instrument_factory(make)()
        self.assertEqual(env.seed_manager.seed_list, [4, 2])
        self.assertEqual(env.seed_manager.ed_idx, 2)
        self.assertEqual(make().seed_manager.seed_list, [0, 2, 4])

    def test_gripper_snapshot_copies_and_invalidates_on_physics_and_exception(self):
        class Environment:
            def __init__(self):
                self.value = 1
                self.reads = 0
                self.robot = object()
                self.robot_manager = SimpleNamespace(get_end_effector_real_val=self.get)

            def get(self, robot, env_idx_list=None):
                self.reads += 1
                return {
                    i: np.array([self.value]) if env_idx_list is None or i in env_idx_list else None for i in range(2)
                }

            def sim_step(self):
                self.value += 1

            def step(self):
                read = self.robot_manager.get_end_effector_real_val
                first = read(self.robot, env_idx_list=[0])
                first[0][0] = 99
                self.observed = read(self.robot, env_idx_list=[0])[0][0]
                self.sim_step()
                self.after_physics = read(self.robot)[0][0]
                raise RuntimeError("control interrupted")

        env = Environment()
        install(env, enabled=True, shadow_epochs=64)
        with self.assertRaisesRegex(RuntimeError, "control interrupted"):
            env.step()
        self.assertEqual(env.observed, 1)
        self.assertEqual(env.after_physics, 2)
        self.assertGreater(env._gripper_snapshot_stats["shadow_calls"], 0)
        env.value = 3
        self.assertEqual(env.robot_manager.get_end_effector_real_val(env.robot)[0][0], 3)

    def test_client_entry_preserves_hook_import_path_for_native_reexec(self):
        entry = ROOT / "runtime/instrument_entry.py"
        captured = {}

        def launch(path, run_name):
            captured.update(path=path, args=list(sys.argv), pythonpath=os.environ["PYTHONPATH"])

        with (
            patch("runpy.run_path", side_effect=launch),
            patch.object(sys, "argv", [str(entry), "--task_name", "play_tic_tac_toe"]),
            patch.object(sys, "path", list(sys.path)),
            patch.dict(os.environ, PYTHONPATH="/existing/import/path"),
        ):
            exec(compile(entry.read_text(), str(entry), "exec"), {"__file__": str(entry)})
        self.assertEqual(captured["args"][-2:], ["--env_factory_module", "client_runtime"])
        self.assertEqual(captured["args"][0], captured["path"])
        self.assertEqual(captured["pythonpath"].split(os.pathsep), [str(ROOT / "runtime"), "/existing/import/path"])

    def test_camera_fence_delivers_twice_and_restores_settings_on_error(self):
        import counter_delivery

        flag = "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges"
        counter = "/rtx/externalFrameCounter"
        values = {
            "/rtx/rendermode": "PathTracing",
            "/rtx/pathtracing/spp": 16,
            "/rtx/pathtracing/totalSpp": 16,
            "/rtx/pathtracing/optixDenoiser/enabled": True,
        }
        before = dict(values)
        settings = SimpleNamespace(
            get=values.get,
            set=lambda k, v: values.__setitem__(k, v),
            set_bool=lambda k, v: values.__setitem__(k, v),
            set_int=lambda k, v: values.__setitem__(k, v),
            destroy_item=lambda k: values.pop(k, None),
        )

        class Context:
            current_time = 4.0
            stamp = 0

            def add_physics_callback(self, name, callback):
                self.callback = callback

            def remove_physics_callback(self, name):
                del self.callback

            def render(self):
                self.stamp += 1

        context = Context()
        carb_settings = SimpleNamespace(get_settings=lambda: settings)
        modules = {
            "carb": SimpleNamespace(settings=carb_settings),
            "carb.settings": carb_settings,
            "env.camera_manager.capture.render_sync": SimpleNamespace(dispatcher_time=lambda: context.stamp),
        }

        def native(ctx, min_passes):
            self.assertTrue(values[flag])
            self.assertGreater(values[counter], 0)
            for _ in range(min_passes):
                ctx.render()
            return min_passes

        def broken(ctx, min_passes):
            ctx.render()
            raise RuntimeError("render failed")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, PERF_WORKER_OUT=tmp),
        ):
            self.assertEqual(counter_delivery.capture_fence(context, native), 2)
            self.assertEqual(context.stamp, 2)
            self.assertEqual(values, before)
            self.assertFalse(hasattr(context, "callback"))
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                counter_delivery.capture_fence(context, broken)
            self.assertEqual(values, before)
            self.assertFalse(hasattr(context, "callback"))
            self.assertEqual(context.current_time, 4.0)
            context.render()
            self.assertEqual(context.stamp, 4)

    def test_prompt_cache_eviction_keeps_lru_order_without_iteration_mutation(self):
        class Cache(OrderedDict):
            _maxsize = 32
            _evict_warned = False

            def __getitem__(self, key):
                self.move_to_end(key)
                return super().__getitem__(key)

        repair_prompt_cache(
            SimpleNamespace(_BoundedPromptEmbedCache=Cache, logger=SimpleNamespace(warning=lambda *a: None))
        )
        cache = Cache()
        for i in range(768):
            cache[str(i)] = i
        self.assertEqual(list(cache), list(map(str, range(736, 768))))
        self.assertEqual(cache["736"], 736)
        cache["768"] = 768
        self.assertIn("736", cache)
        self.assertNotIn("737", cache)
        self.assertEqual(len(cache), 32)

    def test_policy_history_is_isolated_between_workers(self):
        # Exercise the actual adapter without importing Isaac Sim/OpenWAM.
        tree = ast.parse((ROOT / "runtime/server_entry.py").read_text())
        functions = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "scoped_call"]
        identity = contextvars.ContextVar("test_worker")
        profile = SimpleNamespace(ACTIVE=contextvars.ContextVar("profile_worker"), install=lambda model: None)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(sys.modules, profile_policy=profile),
            patch.dict(os.environ, OW_SERVER_METRICS=str(Path(tmp) / "server.jsonl")),
        ):
            import time

            namespace = dict(identity=identity, time=time, os=os, json=json)
            exec(compile(ast.Module(body=functions, type_ignores=[]), "<policy adapter>", "exec"), namespace)

            class Server:
                def __init__(self):
                    self._model_lock = asyncio.Lock()
                    self.model = SimpleNamespace(_batch={}, _order=[])

                async def _invoke_method(self, method, ids):
                    count = self.model._batch.get("calls", 0) + 1
                    self.model._batch["calls"] = count
                    return [[count] * 32 for _ in ids]

            def get_action_batch():
                pass

            async def exercise():
                server = Server()
                result = []
                for worker in ["a", "b", "a"]:
                    identity.set(worker)
                    value = await namespace["scoped_call"](server, get_action_batch, [0])
                    result.append(value[0][0])
                return result

            self.assertEqual(asyncio.run(exercise()), [1, 1, 2])


if __name__ == "__main__":
    unittest.main()
