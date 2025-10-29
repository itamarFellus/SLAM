# %%
# main.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from SLAM.maps.map import MapParams, make_world_grid, world_to_map, map_to_world
from core.registry import make
import plugins  # noqa: F401 ensure registrations


def run() -> None:
    # --- load config ---
    cfg_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- params ---
    mparams = MapParams(**(cfg.get("map", {}) or {}))

    # --- world & mapper ---
    world = make_world_grid(mparams)
    mapper_cfg = cfg.get("mapper", {})
    mapper = make(mapper_cfg["type"], **(mapper_cfg.get("params", {}) or {}))

    # --- agent & sensor ---
    agent_cfg = cfg.get("agent", {})
    sensor_cfg = cfg.get("sensor", {})
    agent = make(agent_cfg["type"], **(agent_cfg.get("params", {}) or {}))
    sensor = make(sensor_cfg["type"], **(sensor_cfg.get("params", {}) or {}))

    # --- start pose (meters, radians) ---
    sim = cfg.get("simulation", {})
    start_row = sim.get("start_row", 140)
    start_col = sim.get("start_col", 20)
    x0, y0 = map_to_world(start_row, start_col, mparams)
    pose = (x0, y0, math.radians(0.0))

    # --- logs ---
    gt_path = [pose]

    # --- simulation loop ---
    steps = int(sim.get("steps", 400))
    schedule = sim.get("steering_program", [])
    idx = 0
    for k in range(steps):
        # pick v,w based on schedule
        while idx < len(schedule) and k >= schedule[idx]["until"]:
            idx += 1
        if idx < len(schedule):
            v = float(schedule[idx].get("v", 0.0))
            w = float(schedule[idx].get("w", 0.0))
        else:
            v = 0.0
            w = 0.0

        # ground-truth evolve
        pose = agent.step(pose, v, w)
        gt_path.append(pose)

        # sensor reading
        ranges, angles_used = sensor.read(world, pose)

        # mapping update using the actual angles used by the sensor
        max_range_m = sensor.params.max_range_m if hasattr(sensor, "params") else 15.0
        mapper.update_with_scan(pose, ranges, angles_used, max_range_m)

    # --- outputs ---
    os.makedirs('/results/plots' , exist_ok=True)

    # Plot learned occupancy grid probabilities
    prob = mapper.to_prob()
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