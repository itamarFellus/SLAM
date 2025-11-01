from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import math


@dataclass
class EKFProcessNoise:
    """Additive process noise standard deviations on the state [x, y, yaw]."""
    sigma_x: float = 0.03
    sigma_y: float = 0.03
    sigma_yaw: float = 0.01


@dataclass
class EKFInit:
    dt: float = 0.2
    init_cov_diag: Tuple[float, float, float] = (0.1, 0.1, math.radians(5.0))
    process_noise: EKFProcessNoise = field(default_factory=EKFProcessNoise)


class EKFLocalization:
    """
    Extended Kalman Filter for 2D unicycle robot pose [x, y, yaw].

    - Motion model: x_{k+1} = x_k + v cos(yaw) dt; y_{k+1} = y_k + v sin(yaw) dt; yaw_{k+1} = yaw_k + w dt
    - Measurement model (default): direct noisy pose measurement z = [x, y, yaw] + noise

    You can use update_pose() with any pose-like measurement (e.g., scan-matching output)
    by supplying its covariance R (3x3) to fuse with the estimate.
    """

    def __init__(self, init_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0), cfg: EKFInit | None = None) -> None:
        self.cfg = cfg or EKFInit()
        self.x = np.array(init_pose, dtype=float).reshape(3, 1)
        self.P = np.diag(np.square(np.array(self.cfg.init_cov_diag, dtype=float)))

        q = self.cfg.process_noise
        self.Q = np.diag([q.sigma_x**2, q.sigma_y**2, q.sigma_yaw**2])

    def predict(self, v: float, w: float) -> None:
        dt = self.cfg.dt
        x, y, yaw = float(self.x[0]), float(self.x[1]), float(self.x[2])
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Nonlinear motion
        x_pred = x + v * cos_y * dt
        y_pred = y + v * sin_y * dt
        yaw_pred = self._wrap_angle(yaw + w * dt)
        self.x = np.array([x_pred, y_pred, yaw_pred], dtype=float).reshape(3, 1)

        # Jacobian of motion w.r.t state
        F = np.array(
            [
                [1.0, 0.0, -v * sin_y * dt],
                [0.0, 1.0,  v * cos_y * dt],
                [0.0, 0.0,  1.0],
            ],
            dtype=float,
        )

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    def update_pose(self, z_pose: Tuple[float, float, float], R: np.ndarray) -> None:
        """Fuse a direct pose measurement z with covariance R (3x3)."""
        z = np.array(z_pose, dtype=float).reshape(3, 1)
        H = np.eye(3)
        y = z - H @ self.x
        y[2, 0] = self._wrap_angle(y[2, 0])  # normalize yaw residual

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P
        self.x[2, 0] = self._wrap_angle(self.x[2, 0])

    def get_pose(self) -> Tuple[float, float, float]:
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi


