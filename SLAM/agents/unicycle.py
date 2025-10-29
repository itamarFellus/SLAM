from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from interfaces.agent import Agent
from SLAM.agent.lidar_based import MotionParams as _MotionParams, step_motion as _step_motion


@dataclass
class UnicycleMotionConfig:
    v_nom: float = 0.4
    w_nom: float = 0.3
    step_dt: float = 0.2


class UnicycleAgent(Agent):
    def __init__(self, motion: UnicycleMotionConfig | None = None):
        self.motion = motion or UnicycleMotionConfig()
        self._delegate = _MotionParams(
            v_nom=self.motion.v_nom,
            w_nom=self.motion.w_nom,
            step_dt=self.motion.step_dt,
        )
        self._pose: Tuple[float, float, float] | None = None

    def set_pose(self, pose: Tuple[float, float, float]) -> None:
        self._pose = pose

    def step(self, pose: Tuple[float, float, float], v: float | None = None, w: float | None = None) -> Tuple[float, float, float]:
        v_use = self._delegate.v_nom if v is None else v
        w_use = self._delegate.w_nom if w is None else w
        new_pose = _step_motion(pose, v_use, w_use, self._delegate.step_dt)
        self._pose = new_pose
        return new_pose

    def pose(self) -> Tuple[float, float, float] | None:
        return self._pose


