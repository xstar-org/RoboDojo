import ast
from collections.abc import Iterable, Mapping
from copy import deepcopy
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List
import unittest
from unittest.mock import patch


class LayoutShardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name) / "Eval_Layout/RoboDojo/test/1"
        directory.mkdir(parents=True)
        for i in range(10):
            (directory / f"cover_blocks_{i}.json").write_text("{}")
        path = Path(__file__).parents[1] / "env/seed_manager/seed_manager.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SeedManager")
        ns = dict(globals(), ASSETS_PATH=self.temp.name, BENCHMARK="RoboDojo")
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
        self.manager = ns["SeedManager"](dict(num_envs=5, task_name="cover_blocks", config_name="test", seed=1))

    def test_disjoint_shards_keep_original_layout_files(self):
        found = []
        for shard in ("0,1,2,3,4", "5,6,7,8,9"):
            with patch.dict(os.environ, ROBODOJO_EVAL_LAYOUT_IDS=shard):
                self.manager.init_eval()
            batch = self.manager.get_seeds(max_count=5)
            found.extend(batch)
            for i in batch:
                self.assertTrue(self.manager.seed_info[i]["scene_layout"].endswith(f"cover_blocks_{i}.json"))
            self.assertIsNone(self.manager.get_seeds(max_count=5))
        self.assertEqual(found, list(range(10)))

    def test_resume_filter_applies_within_shard(self):
        with patch.dict(os.environ, ROBODOJO_EVAL_LAYOUT_IDS="5,6,7,8,9"):
            self.manager.init_eval(completed_layout_ids=[5, 0], abandoned_layout_ids=[8])
        self.assertEqual(self.manager.get_seeds(max_count=5), [6, 7, 9])

    def test_invalid_shards_are_rejected(self):
        for shard in ("", "1,1", "-1", "10", "a", "1,"):
            with self.subTest(shard=shard), patch.dict(os.environ, ROBODOJO_EVAL_LAYOUT_IDS=shard):
                with self.assertRaises(ValueError):
                    self.manager.init_eval()

    def test_default_keeps_all_layouts(self):
        with patch.dict(os.environ):
            os.environ.pop("ROBODOJO_EVAL_LAYOUT_IDS", None)
            self.manager.init_eval()
        self.assertEqual(self.manager.seed_list, list(range(10)))


if __name__ == "__main__":
    unittest.main()
