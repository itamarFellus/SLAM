from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np


class Planner(ABC):
    @abstractmethod
    def plan(self, world: np.ndarray, start_pose: Tuple[float, float, float], goal_pose: Tuple[float, float, float]) -> List[Tuple[float, float]]:
        ...


