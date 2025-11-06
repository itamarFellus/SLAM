from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from interfaces.sensor import Sensor
from SLAM.agent.lidar_based import (
    LidarParams as _LidarParams,
    simulate_lidar_scan as _simulate_lidar_scan,
)
from SLAM.maps.map import MapParams


@dataclass
class LidarConfig:
    fov_deg: float = 270.0
    num_beams: int = 180
    max_range_m: float = 15.0
    sigma_r: float = 0.02
    dropout_prob: float = 0.02
    angle_noise_deg: float = 0.1


class LidarSensor(Sensor):
    def __init__(self, params: LidarConfig | None = None, map_params: MapParams | None = None):
        self.params = params or LidarConfig()
        self.map_params = map_params or MapParams()
        self._delegate = _LidarParams(
            fov_deg=self.params.fov_deg,
            num_beams=self.params.num_beams,
            max_range_m=self.params.max_range_m,
            sigma_r=self.params.sigma_r,
            dropout_prob=self.params.dropout_prob,
            angle_noise_deg=self.params.angle_noise_deg,
        )

    def read(self, world: np.ndarray, pose: Tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        return _simulate_lidar_scan(world, pose, self._delegate, self.map_params)


