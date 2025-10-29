from __future__ import annotations

from typing import Tuple
import numpy as np

from interfaces.mapper import Mapper
from SLAM.maps.map import MapParams, OccupancyGrid as _OccupancyGrid


class OccupancyGridMapper(Mapper):
    def __init__(self, map_params: MapParams | None = None):
        self.map_params = map_params or MapParams()
        self._delegate = _OccupancyGrid(self.map_params)

    def update_with_scan(
        self,
        pose: Tuple[float, float, float],
        ranges: np.ndarray,
        angles: np.ndarray,
        max_range_m: float,
    ) -> None:
        self._delegate.update_with_scan(pose, ranges, angles, max_range_m)

    def to_prob(self) -> np.ndarray:
        return self._delegate.to_prob()


