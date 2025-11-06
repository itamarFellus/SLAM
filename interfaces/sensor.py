from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Sensor(ABC):
    @abstractmethod
    def read(self, world: np.ndarray, pose: Tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        ...


