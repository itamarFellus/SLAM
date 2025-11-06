from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import numpy as np


class Reward(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def compute(
        self,
        mapper: Any,
        world: np.ndarray,
        pose: Tuple[float, float, float],
        ranges: np.ndarray,
        angles: np.ndarray,
        collision_detected: bool,
    ) -> Tuple[float, Dict[str, float]]:
        ...


