"""BBox access must return independent vertices without copying unrelated metadata."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class BboxMetadataTests(unittest.TestCase):
    def manager(self, vertices):
        source = Path(__file__).parents[1] / "env/scene_manager/layout_manager.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LayoutManager")
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_instance_bbox_vertices"]
        namespace = {"np": np}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
        manager = namespace["LayoutManager"]()
        manager.get_scene_object = lambda *args: object()
        manager.instance_type_by_env = [{"box": "rigid"}]
        metadata = {"geometry": {"oriented_bbox": {"vertices": vertices}}}
        manager.object_records_by_type = {"Rigid": SimpleNamespace(metadata_by_env=[{"box": metadata}])}
        return manager, metadata

    def test_list_and_array_vertices_are_independent_and_fresh(self):
        for vertices in ([[1, 2, 3], [4, 5, 6]], np.arange(6.0).reshape(2, 3)):
            manager, metadata = self.manager(vertices)
            expected = np.array(vertices, dtype=float, copy=True)
            result = manager.get_instance_bbox_vertices("box", 0)
            np.testing.assert_array_equal(result, expected)
            result[:] = -1
            np.testing.assert_array_equal(vertices, expected)
            metadata["geometry"]["oriented_bbox"]["vertices"] = [[9, 8, 7]]
            np.testing.assert_array_equal(manager.get_instance_bbox_vertices("box", 0), [[9, 8, 7]])

    def test_fallback_and_missing_object(self):
        manager, metadata = self.manager([[0, 0, 0]])
        metadata.clear()
        fallback = np.ones((8, 3))
        manager.get_scene_object = lambda *args: SimpleNamespace(get_bbox=lambda: fallback)
        self.assertIs(manager.get_instance_bbox_vertices("box", 0), fallback)
        manager.get_scene_object = lambda *args: None
        self.assertIsNone(manager.get_instance_bbox_vertices("box", 0))


if __name__ == "__main__":
    unittest.main()
