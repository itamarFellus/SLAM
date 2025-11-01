from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ExplorerConfig:
    v_fwd: float = 0.4
    w_gain: float = 1.2
    min_clearance_m: float = 0.6


class ReactiveExplorer:
    def __init__(self, cfg: ExplorerConfig | None = None) -> None:
        self.cfg = cfg or ExplorerConfig()

    def next_control(self, ranges: np.ndarray, angles_used: np.ndarray) -> Tuple[float, float]:
        # Improved policy: avoid obstacles, explore open spaces, prevent getting stuck
        if ranges.size == 0:
            return 0.0, 0.0
        
        # Find closest obstacle
        i_min = int(np.argmin(ranges))
        r_min = float(ranges[i_min])
        ang_min = float(angles_used[i_min])
        
        # Velocity control: slow down near obstacles
        if r_min < self.cfg.min_clearance_m * 0.5:
            # Very close - stop forward motion
            v = 0.0
        elif r_min < self.cfg.min_clearance_m:
            # Close - slow down significantly
            v = 0.3 * self.cfg.v_fwd
        else:
            # Clear - full speed ahead
            v = self.cfg.v_fwd
        
        # Angular velocity control: steer away from obstacles
        if r_min < self.cfg.min_clearance_m:
            # Turn away from closest obstacle
            # Use a smoother turning response
            clearance_violation = max(0.0, self.cfg.min_clearance_m - r_min)
            # Turn in the opposite direction of the obstacle
            # Use sin(ang) to determine turn direction (negative = left, positive = right)
            w = -self.cfg.w_gain * np.sign(np.sin(ang_min)) * (1.0 + clearance_violation)
        else:
            # When clear, explore by turning towards the direction with most clearance
            # Find the direction with maximum range (most open space)
            i_max = int(np.argmax(ranges))
            ang_max = float(angles_used[i_max])
            r_max = float(ranges[i_max])
            
            # Turn towards open space, but only if it's significantly more open
            if r_max > r_min * 1.5:
                # Gentle turn towards open space
                w = 0.3 * self.cfg.w_gain * np.sign(np.sin(ang_max))
            else:
                # Minimal exploration turn to prevent getting stuck
                w = 0.1 * self.cfg.w_gain
        
        return v, w


