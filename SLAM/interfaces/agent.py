from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Protocol, Any


class Agent(ABC):
    @abstractmethod
    def step(self, pose: Tuple[float, float, float], v: float | None = None, w: float | None = None) -> Tuple[float, float, float]:
        ...

    @abstractmethod
    def pose(self) -> Tuple[float, float, float] | None:
        ...


