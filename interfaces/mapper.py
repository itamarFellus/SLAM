from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Mapper(ABC):
    @abstractmethod
    def update_with_scan(
        self,
        pose: Tuple[float, float, float],
        ranges: np.ndarray,
        angles: np.ndarray,
        max_range_m: float,
    ) -> None:
        ...

    @abstractmethod
    def to_prob(self) -> np.ndarray:
        ...


