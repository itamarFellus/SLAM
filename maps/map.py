# map.py
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Iterable

@dataclass
class MapParams:
    map_size: Tuple[int, int] = (200, 200)    # (rows, cols) in cells
    map_resolution: float = 0.1               # meters per cell
    # log-odds parameters
    lo_occ: float = 0.85                      # occupied boost per hit
    lo_free: float = 0.4                      # free-space decrement per miss
    lo_min: float = -4.0
    lo_max: float = 4.0


def make_world_grid(params: MapParams) -> np.ndarray:
    h, w = params.map_size
    grid = np.zeros((h, w), dtype=np.uint8)  # 0=free, 1=occupied
    # Add border walls
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    # Add a few rectangles
    def rect(y0, x0, y1, x1):
        grid[y0:y1, x0:x1] = 1
    rect(40, 30, 120, 35)
    rect(80, 90, 85, 170)
    rect(130, 30, 135, 150)
    rect(30, 150, 160, 155)
    rect(150, 60, 170, 140)
    return grid

# -------------------------------
# Utilities
# -------------------------------

def world_to_map(x: float, y: float, params: MapParams) -> Tuple[int, int]:
    """Convert world meters (x right, y up) to map indices (row, col) from top-left."""
    col = int(x / params.map_resolution)
    row = int((params.map_size[0] * params.map_resolution - y) / params.map_resolution)
    return row, col


def map_to_world(row: int, col: int, params: MapParams) -> Tuple[float, float]:
    x = (col + 0.5) * params.map_resolution
    y = params.map_size[0] * params.map_resolution - (row + 0.5) * params.map_resolution
    return x, y


def bresenham(r0: int, c0: int, r1: int, c1: int) -> Iterable[Tuple[int, int]]:
    """Grid cells along a line using Bresenham's algorithm (generator)."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = (dr - dc)
    r, c = r0, c0
    while True:
        yield (r, c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


class OccupancyGrid:
    """Log-odds occupancy grid that is agnostic to the specific sensor.

    update_with_scan() takes the sensor pose, *the actual per-beam angles used*,
    the measured ranges, and the sensor's maximum range. This avoids coupling the
    mapper to any particular LiDAR configuration.
    """

    def __init__(self, params: MapParams):
        self.params = params
        h, w = params.map_size
        self.log_odds = np.zeros((h, w), dtype=np.float32)
        self.seen = np.zeros((h, w), dtype=bool)

    def update_with_scan(
        self,
        pose: Tuple[float, float, float],
        ranges: np.ndarray,
        angles: np.ndarray,
        max_range_m: float,
    ) -> None:
        x, y, yaw = pose
        h, w = self.log_odds.shape
        lo_free = self.params.lo_free
        lo_occ = self.params.lo_occ
        
        # Pre-compute cos/sin for all angles (vectorized)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        for r, ang, cos_a, sin_a in zip(ranges, angles, cos_angles, sin_angles):
            # end point in world meters for this beam
            end_x = x + r * cos_a
            end_y = y + r * sin_a

            r0, c0 = world_to_map(x, y, self.params)
            r1, c1 = world_to_map(end_x, end_y, self.params)

            # Collect cells from generator, processing efficiently
            cells_list = []
            last_cell = None
            for cell in bresenham(r0, c0, r1, c1):
                cells_list.append(cell)
                last_cell = cell
            
            if len(cells_list) == 0:
                continue
            
            # Mark all traversed cells (including endpoint) as seen
            seen_cells = np.array(cells_list, dtype=np.int32)
            rows_all, cols_all = seen_cells[:, 0], seen_cells[:, 1]
            valid_all = (rows_all >= 0) & (rows_all < h) & (cols_all >= 0) & (cols_all < w)
            if np.any(valid_all):
                self.seen[rows_all[valid_all], cols_all[valid_all]] = True
                
            # Batch process free cells (all except last) using numpy
            if len(cells_list) > 1:
                free_cells = np.array(cells_list[:-1], dtype=np.int32)
                rows, cols = free_cells[:, 0], free_cells[:, 1]
                # Vectorized bounds checking
                valid_mask = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
                valid_rows = rows[valid_mask]
                valid_cols = cols[valid_mask]
                # Batch update using numpy fancy indexing
                if len(valid_rows) > 0:
                    self.log_odds[valid_rows, valid_cols] -= lo_free
            
            # Occupied at the hit (if within max range and not a "no return")
            if r < max_range_m * 0.999 and last_cell is not None:
                rr, cc = last_cell
                if 0 <= rr < h and 0 <= cc < w:
                    self.log_odds[rr, cc] += lo_occ

        # clamp
        np.clip(self.log_odds, self.params.lo_min, self.params.lo_max, out=self.log_odds)

    def to_prob(self) -> np.ndarray:
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def get_seen_mask(self) -> np.ndarray:
        return self.seen.copy()