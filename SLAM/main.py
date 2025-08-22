# %%
# main.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from SLAM.maps.map import MapParams, make_world_grid, world_to_map, map_to_world, OccupancyGrid
from SLAM.agent.lidar_based import LidarParams, MotionParams, LidarAgent, simulate_lidar_scan


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
    os.makedirs('/results/plots' , exist_ok=True)

    # Plot learned occupancy grid probabilities
    prob = og.to_prob()
    plt.figure()
    plt.imshow(prob, origin="upper", cmap="gray")
    plt.title("Learned Occupancy Grid (probabilities)")
    plt.xlabel("x (cells)")
    plt.ylabel("y (cells)")
    plt.tight_layout()
    plt.show()
    # plt.savefig(out_dir / "occupancy_grid_final.png", dpi=150)
    # plt.close()

    # Plot trajectories over ground-truth world for sanity
    disp = np.ones_like(world, dtype=float)
    disp[world == 1] = 0.2

    xs_gt = [p[0] for p in gt_path]
    ys_gt = [p[1] for p in gt_path]

    pts_gt = np.array([world_to_map(x, y, mparams) for x, y in zip(xs_gt, ys_gt)])

    plt.figure()
    plt.imshow(disp, origin="upper", cmap="gray")
    plt.plot(pts_gt[:, 1], pts_gt[:, 0], label="GT path")
    plt.legend()
    plt.title("Trajectory over World (for sanity check)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()