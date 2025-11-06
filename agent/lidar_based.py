# lidar_based.py
import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from maps.map import world_to_map, MapParams


@dataclass
class LidarParams:
    fov_deg: float = 270.0
    num_beams: int = 180
    max_range_m: float = 15.0
    sigma_r: float = 0.02            # range noise (m)
    dropout_prob: float = 0.02
    angle_noise_deg: float = 0.1


@dataclass
class MotionParams:
    v_nom: float = 0.4               # linear velocity (m/s)
    w_nom: float = 0.3               # angular velocity (rad/s)
    step_dt: float = 0.2             # seconds


def step_motion(pose: Tuple[float, float, float], v: float, w: float, dt: float) -> Tuple[float, float, float]:
    x, y, yaw = pose
    # unicycle model
    x += v * math.cos(yaw) * dt
    y += v * math.sin(yaw) * dt
    yaw += w * dt
    # wrap to [-pi, pi)
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    return (x, y, yaw)


def raycast(world: np.ndarray, x: float, y: float, theta: float, map_params: MapParams, max_range_m: float) -> float:
    """Cast a single ray and return range (meters)."""
    max_steps = int(max_range_m / map_params.map_resolution)
    dx = math.cos(theta) * map_params.map_resolution
    dy = math.sin(theta) * map_params.map_resolution
    cx, cy = x, y
    h, w = world.shape
    for _ in range(max_steps):
        cx += dx
        cy += dy
        r, c = world_to_map(cx, cy, map_params)
        if r < 0 or r >= h or c < 0 or c >= w:
            return max_range_m
        if world[r, c] == 1:
            dist = math.hypot(cx - x, cy - y)
            return min(dist, max_range_m)
    return max_range_m


def simulate_lidar_scan(
    world: np.ndarray,
    pose: Tuple[float, float, float],
    lidar: LidarParams,
    map_params: MapParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ranges, angles_used). Angles include per-beam angle noise.
    The caller should use the returned angles for mapping.
    """
    x, y, yaw = pose
    start = yaw - math.radians(lidar.fov_deg) / 2.0
    nominal = start + np.arange(lidar.num_beams) * math.radians(lidar.fov_deg) / max(1, (lidar.num_beams - 1))

    angles_used = nominal + np.random.normal(0.0, math.radians(lidar.angle_noise_deg), size=lidar.num_beams)

    ranges = np.zeros(lidar.num_beams, dtype=np.float32)
    for i, ang in enumerate(angles_used):
        r = raycast(world, x, y, ang, map_params, lidar.max_range_m)
        # dropout
        if np.random.rand() < lidar.dropout_prob:
            r = lidar.max_range_m
        # range noise
        r = r + np.random.normal(0.0, lidar.sigma_r)
        r = np.clip(r, 0.0, lidar.max_range_m)
        ranges[i] = r
    return ranges, angles_used


class LidarAgent:
    """Simple agent that carries LiDAR and moves with a unicycle model."""
    def __init__(self, lidar: LidarParams, motion: MotionParams):
        self.lidar = lidar
        self.motion = motion

    def step(self, pose: Tuple[float, float, float], v: float | None = None, w: float | None = None) -> Tuple[float, float, float]:
        v = self.motion.v_nom if v is None else v
        w = self.motion.w_nom if w is None else w
        return step_motion(pose, v, w, self.motion.step_dt)