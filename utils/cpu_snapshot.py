"""Explicit, coalesced tensor transfer at CPU control/reward boundaries."""

from collections import defaultdict

import numpy as np
import torch


def cpu_snapshot(values):
    """Copy a mapping of arrays to independent host arrays.

    CUDA tensors of the same device/dtype share one transfer, regardless of
    environment count. Snapshots are owned by the caller and never cached here;
    a simulation step or reset therefore cannot leave hidden stale state.
    """
    result = {}
    groups = defaultdict(list)
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            groups[(value.device, value.dtype)].append((name, value.detach()))
        else:
            result[name] = np.array(value, copy=True)
    for entries in groups.values():
        flat = torch.cat([value.reshape(-1) for _, value in entries])
        host = flat.cpu().numpy()
        offset = 0
        for name, value in entries:
            size = value.numel()
            result[name] = host[offset : offset + size].reshape(tuple(value.shape))
            offset += size
    return result
