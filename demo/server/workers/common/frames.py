"""Frame conversion without server transport concerns."""

from __future__ import annotations

import numpy as np


def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """Normalize grayscale/CHW/HWC and float/uint8 frames to HWC RGB."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"unsupported frame shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(arr.min()) < -0.01:
            arr = (arr + 1.0) * 127.5
        elif not arr.size or float(arr.max()) <= 1.001:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, :3])
