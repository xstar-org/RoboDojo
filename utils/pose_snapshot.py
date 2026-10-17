"""Convert frozen world poses together for CPU reward evaluation."""

import numpy as np
from pxr import Gf
from scipy.spatial.transform import Rotation


def local_poses_from_world(parent_transforms, positions, orientations):
    """Use Isaac's float32 matrix and Gf decomposition conventions.

    USD matrices use row vectors. World poses are wxyz; the same SciPy
    normalization and Gf rotation extraction as Isaac preserve parent scale
    and rotation, rather than assuming that environments only translate.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    orientations = np.asarray(orientations, dtype=np.float32).reshape(-1, 4)
    parents = np.asarray(parent_transforms, dtype=np.float32).reshape(-1, 4, 4)
    world = np.zeros((len(positions), 4, 4), dtype=np.float32)
    world[:, :3, :3] = Rotation.from_quat(orientations[:, [1, 2, 3, 0]]).as_matrix()
    world[:, :3, 3] = positions
    world[:, 3, 3] = 1
    local = np.linalg.solve(parents.transpose(0, 2, 1), world)
    translations = np.empty_like(positions)
    rotations = np.empty_like(orientations)
    for index, matrix in enumerate(local):
        transform = Gf.Transform()
        transform.SetMatrix(Gf.Matrix4d(matrix.T.tolist()))
        translations[index] = transform.GetTranslation()
        quat = transform.GetRotation().GetQuat()
        rotations[index] = [quat.GetReal(), *quat.GetImaginary()]
    return translations, rotations
