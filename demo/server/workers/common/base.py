"""Framework-neutral world-model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .actions import Action


class WorldModel(ABC):
    """Stateful one-step world model used by both server and research code."""

    fps: int = 20
    quality: int = 80

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset an episode and return its first frame."""

    @abstractmethod
    def step(self, action: Action) -> np.ndarray:
        """Apply one action and return the next frame."""

    def close(self) -> None:
        """Release model resources, if any."""

    def info(self) -> dict[str, Any]:
        return {"fps": self.fps, "quality": self.quality}
