import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "usd_writeback", Path(__file__).resolve().parents[1] / "env/environment/usd_writeback.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Settings:
    def __init__(self):
        self.value = True

    def get(self, key):
        assert key == module.USD_WRITEBACK
        return self.value

    def set(self, key, value):
        assert key == module.USD_WRITEBACK
        self.value = value


class Task:
    def step(self):
        pass


class Queue:
    def __init__(self, count=0):
        self.count = count

    def is_empty(self):
        return self.count == 0


class World(Task):
    def __init__(self, fail_at=None):
        self.settings = Settings()
        self.queues = [Queue() for _ in range(4)]
        self.sim = SimpleNamespace(
            sim=SimpleNamespace(device="cpu"),
            event_manager=SimpleNamespace(available_modes=[]),
        )
        self.sim_config = {"defer_usd_writeback": True}
        self.use_fabric = False
        self.robot_manager = SimpleNamespace(
            control_manager=SimpleNamespace(control_queue=self.queues)
        )
        self.native = self.usd = 0
        self.flags = []
        self.fail_at = fail_at

    def sim_step(self, render=False):
        self.native += 1
        self.flags.append(self.settings.value)
        if self.settings.value:
            self.usd = self.native
        if self.native == self.fail_at:
            raise ValueError("native physics failed")

    def run(self, ids=(0, 1, 2, 3), groups=10):
        for index in ids:
            self.queues[index].count = groups
        with module.deferred_usd_writeback(self, ids):
            while ids and not any(self.queues[i].is_empty() for i in ids):
                for index in ids:
                    self.queues[index].count -= 1
                self.sim_step(render=False)
        assert self.usd == self.native, "consumer observed stale USD"


class WritebackTests(unittest.TestCase):
    def setUp(self):
        self.world = World()
        carb = ModuleType("carb")
        carb.settings = ModuleType("carb.settings")
        carb.settings.get_settings = lambda: self.world.settings
        task = ModuleType("env.environment.task_env")
        task.TaskEnv = Task
        self.modules = patch.dict(sys.modules, {
            "carb": carb, "carb.settings": carb.settings,
            "env.environment.task_env": task,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_retains_all_steps_and_publishes_before_consumers(self):
        self.world.run()
        self.assertEqual(self.world.native, 10)
        self.assertEqual(self.world.flags, [False] * 9 + [True])
        self.assertEqual(self.world.usd_writeback_stats["suppressed_groups"], 9)
        self.assertNotIn("sim_step", vars(self.world))

    def test_default_has_original_behavior_and_no_instance_hooks(self):
        self.world.sim_config = {}
        self.world.run()
        self.assertEqual(self.world.flags, [True] * 10)
        self.assertFalse(hasattr(self.world, "usd_writeback_stats"))
        self.assertNotIn("sim_step", vars(self.world))

    def test_explicit_false_keeps_original_behavior(self):
        self.world.sim_config["defer_usd_writeback"] = False
        self.world.run()
        self.assertEqual(self.world.flags, [True] * 10)
        self.assertFalse(hasattr(self.world, "usd_writeback_stats"))

    def test_partial_ids_do_not_clear_inactive_queues(self):
        self.world.queues[1].count = 7
        self.world.run(ids=(3, 0))
        self.assertEqual(self.world.queues[1].count, 7)
        self.assertEqual(self.world.usd_writeback_stats["published_groups"], 1)

    def test_one_group_and_empty_scope_do_not_leave_stale_state(self):
        self.world.run(groups=1)
        self.world.run(ids=())
        self.assertEqual(self.world.flags, [True])
        self.assertTrue(self.world.settings.value)

    def test_native_exception_restores_settings_and_original_dispatch(self):
        self.world.fail_at = 4
        with self.assertRaisesRegex(ValueError, "native physics failed"):
            self.world.run()
        self.assertEqual(self.world.native, 4)
        self.assertTrue(self.world.settings.value)
        self.assertNotIn("sim_step", vars(self.world))
        self.assertFalse(self.world._usd_writeback_active)
        self.assertEqual(self.world.usd_writeback_stats["exceptions"], 1)

    def test_incomplete_queue_is_not_silently_repaired(self):
        self.world.queues[0].count = 2
        with self.assertRaisesRegex(RuntimeError, "final USD publish"):
            with module.deferred_usd_writeback(self.world, [0]):
                self.world.queues[0].count -= 1
                self.world.sim_step()
        self.assertTrue(self.world.settings.value)
        self.assertEqual(self.world.native, 1)

    def test_preserves_an_existing_instance_dispatch(self):
        original = self.world.sim_step
        override = lambda **kwargs: original(**kwargs)
        self.world.sim_step = override
        self.world.run()
        self.assertIs(self.world.sim_step, override)

    def test_unsupported_consumers_and_backends_keep_original_writeback(self):
        class CustomTask(Task):
            def step(self):
                pass
        for kind in ["cuda", "fabric", "interval", "custom_task"]:
            self.world = type("CustomWorld", (CustomTask, World), {})() if kind == "custom_task" else World()
            if kind == "cuda":
                self.world.sim.sim.device = "cuda:0"
            if kind == "fabric":
                self.world.use_fabric = True
            if kind == "interval":
                self.world.sim.event_manager.available_modes = ["interval"]
            self.world.run()
            self.assertEqual(self.world.flags, [True] * 10, kind)
            self.assertFalse(hasattr(self.world, "usd_writeback_stats"), kind)


if __name__ == "__main__":
    unittest.main()
