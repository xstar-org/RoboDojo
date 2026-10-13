"""Read rigid-body poses once at the CPU reward boundary."""

import numpy as np
import torch

from utils.cpu_snapshot import cpu_snapshot


class RigidPoseReader:
    def __init__(self):
        self.reset()

    def reset(self):
        self._simulation_view = None
        self._signature = None
        self._view = None

    def read(self, objects):
        """Return local CPU poses keyed like ``objects``, or None for fallback.

        Parent transforms are read afresh for every snapshot. Only the PhysX
        view is cached; moving a parent or a body never reuses an old pose.
        """
        if not objects:
            return {}
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.numpy.transformations import get_local_from_world
        from pxr import Usd, UsdGeom

        simulation_view = SimulationManager.get_physics_sim_view()
        if simulation_view is None:
            return None
        by_path = {str(obj.prim_path): (key, obj) for key, obj in objects.items()}
        if len(by_path) != len(objects):
            raise ValueError("Rigid pose snapshot contains duplicate prim paths")
        signature = tuple((path, id(obj)) for path, (_, obj) in by_path.items())
        if simulation_view is not self._simulation_view or signature != self._signature:
            self.reset()
            view = simulation_view.create_rigid_body_view(list(by_path))
            if view is None or set(view.prim_paths) != set(by_path) or len(view.prim_paths) != len(by_path):
                return None
            self._view = view
            self._signature = signature
            self._simulation_view = simulation_view

        transforms = self._view.get_transforms()
        if not isinstance(transforms, torch.Tensor) or not transforms.is_cuda:
            return None
        # Avoid SingleRigidPrim.get_local_pose(): its torch implementation
        # synchronizes to Python lists and creates GPU tensors for every body.
        host = cpu_snapshot({"poses": transforms})["poses"]
        parents = {}
        parent_paths = []
        keys = []
        for path in self._view.prim_paths:
            key, obj = by_path[path]
            parent = obj.stage.GetPrimAtPath(path).GetParent()
            parent_path = str(parent.GetPath())
            if parent_path not in parents:
                parents[parent_path] = np.array(
                    UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                    dtype=np.float32,
                )
            keys.append(key)
            parent_paths.append(parent_path)
        matrices = np.stack([parents[path] for path in parent_paths])
        # PhysX uses xyzw, while the SDK local-pose helper uses wxyz.
        positions, orientations = get_local_from_world(matrices, host[:, :3], host[:, [6, 3, 4, 5]])
        return {key: (positions[i], orientations[i]) for i, key in enumerate(keys)}
