"""Lifecycle and ordering tests; numerical SDK equivalence is checked on GPU."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class Tensor:
    is_cuda = True

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)


def module(**attributes):
    result = ModuleType("stub")
    result.__dict__.update(attributes)
    return result


class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.created = 0
        self.transfers = 0
        self.parents = {"/a": 1.0, "/b": 2.0}
        self.paths = ["/b/body", "/a/body"]
        self.tensor = Tensor([[20, 0, 0, 0, 0, 0, 1], [10, 0, 0, 0, 0, 0, 1]])
        self.simulation = SimpleNamespace(create_rigid_body_view=self.create_view)
        self.objects = {"a": self.object("/a/body"), "b": self.object("/b/body")}

        def snapshot(values):
            self.transfers += 1
            return {key: value.data.copy() for key, value in values.items()}

        # A translation-only SDK stand-in checks the inputs/order and fresh
        # parent reads. Real rotation/scale math is deliberately not reimplemented.
        def local_poses(parents, positions, orientations):
            np.testing.assert_array_equal(orientations, [[1, 0, 0, 0]] * len(positions))
            return positions - parents[:, 3, :3], orientations

        modules = {
            "torch": module(Tensor=Tensor),
            "utils.cpu_snapshot": module(cpu_snapshot=snapshot),
            "isaacsim.core.simulation_manager": module(
                SimulationManager=SimpleNamespace(get_physics_sim_view=lambda: self.simulation)
            ),
            "isaacsim.core.utils.numpy.transformations": module(get_local_from_world=local_poses),
            "pxr": module(
                Usd=SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: None)),
                UsdGeom=SimpleNamespace(Xformable=lambda parent: parent),
            ),
        }
        self.patch = patch.dict(sys.modules, modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        path = Path(__file__).resolve().parents[1] / "utils/rigid_pose_snapshot.py"
        spec = importlib.util.spec_from_file_location("reader_under_test", path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        self.reader = loaded.RigidPoseReader()

    def object(self, path):
        parent_path = path.rsplit("/", 1)[0]

        def transform(time):
            result = np.eye(4)
            result[3, 0] = self.parents[parent_path]
            return result

        parent = SimpleNamespace(GetPath=lambda: parent_path, ComputeLocalToWorldTransform=transform)
        stage = SimpleNamespace(GetPrimAtPath=lambda p: SimpleNamespace(GetParent=lambda: parent))
        return SimpleNamespace(prim_path=path, stage=stage)

    def create_view(self, paths):
        self.created += 1
        return SimpleNamespace(prim_paths=list(self.paths), get_transforms=lambda: self.tensor)

    def test_view_order_and_moving_parent(self):
        poses = self.reader.read(self.objects)
        self.assertEqual(poses["a"][0][0], 9)
        self.assertEqual(poses["b"][0][0], 18)
        self.assertEqual(self.transfers, 1)
        self.parents["/b"] = 7
        self.assertEqual(self.reader.read(self.objects)["b"][0][0], 13)
        self.assertEqual(self.created, 1)
        self.assertEqual(self.transfers, 2)

    def test_rebuild_after_simulation_object_or_layout_reset(self):
        self.reader.read(self.objects)
        self.simulation = SimpleNamespace(create_rigid_body_view=self.create_view)
        self.reader.read(self.objects)
        self.objects["a"] = self.object("/a/body")
        self.reader.read(self.objects)
        self.reader.reset()
        self.reader.read(self.objects)
        self.assertEqual(self.created, 4)

    def test_incomplete_view_falls_back_without_partial_snapshot(self):
        self.paths = ["/a/body"]
        self.assertIsNone(self.reader.read(self.objects))
        self.assertEqual(self.transfers, 0)

    def test_cpu_backend_and_missing_physics_fall_back(self):
        self.tensor.is_cuda = False
        self.assertIsNone(self.reader.read(self.objects))
        self.simulation = None
        self.assertIsNone(self.reader.read(self.objects))
        self.assertEqual(self.transfers, 0)


if __name__ == "__main__":
    unittest.main()
