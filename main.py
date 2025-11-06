# %%
# main.py
import math

import numpy as np

from maps.map import MapParams, make_world_grid, world_to_map, map_to_world, OccupancyGrid
from agent.lidar_based import LidarParams, MotionParams, LidarAgent, simulate_lidar_scan
from plots.figures import (
    plot_occupancy_grid,
    plot_trajectories,
    plot_combined,
    plot_seen_mask,
)


def run() -> None:
    # --- params ---
    mparams = MapParams()
    lparams = LidarParams()
    mot = MotionParams()

    # --- world & mapper ---
    world = make_world_grid(mparams)
    og = OccupancyGrid(mparams)

    # --- agent ---
    agent = LidarAgent(lparams, mot)

    # --- start pose (meters, radians) ---
    start_row, start_col = 140, 20
    x0, y0 = map_to_world(start_row, start_col, mparams)
    pose = (x0, y0, math.radians(0.0))

    # --- logs ---
    gt_path = [pose]

    # --- simulation loop ---
    steps = 400
    for k in range(steps):
        # simple steering program: arc + straights
        if k < 120:
            v = mot.v_nom
            w = mot.w_nom
        elif k < 240:
            v = mot.v_nom
            w = -mot.w_nom * 0.5
        else:
            v = mot.v_nom
            w = 0.0

        # ground-truth evolve
        pose = agent.step(pose, v, w)
        gt_path.append(pose)

        # simulate lidar
        ranges, angles_used = simulate_lidar_scan(world, pose, lparams, mparams)

        # mapping update using the actual angles used by the sensor
        og.update_with_scan(pose, ranges, angles_used, lparams.max_range_m)

    # --- outputs ---
    prob = og.to_prob()
    plot_occupancy_grid(prob)
    plot_trajectories(world, gt_path, None, mparams)
    plot_combined(prob, world, gt_path, None, mparams)
    
    # Plot seen mask
    seen_mask = og.get_seen_mask()
    plot_seen_mask(seen_mask)


if __name__ == "__main__":
    run()