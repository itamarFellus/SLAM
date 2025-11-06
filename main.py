# %%
# main.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
from pathlib import Path
import time

import numpy as np
import yaml

from maps.map import MapParams, make_world_grid, world_to_map, map_to_world
from core.registry import make
import plugins  # noqa: F401 ensure registrations
from plots.figures import (
    plot_total_reward,
    plot_coverage,
    plot_occupancy_grid,
    plot_seen_mask,
    plot_trajectories,
    plot_combined,
)


def check_collision(world: np.ndarray, pose: tuple[float, float, float], mparams: MapParams, 
                    robot_radius_m: float = 0.15) -> bool:
    """Check if a pose would collide with obstacles in the world.
    
    Args:
        world: Grid map (0=free, 1=occupied)
        pose: (x, y, yaw) in world coordinates
        mparams: Map parameters
        robot_radius_m: Radius of the robot in meters (for collision checking)
    
    Returns:
        True if collision detected, False otherwise
    """
    x, y, _ = pose
    r, c = world_to_map(x, y, mparams)
    h, w = world.shape
    
    # Check if robot center is in bounds
    if r < 0 or r >= h or c < 0 or c >= w:
        return True
    
    # Check robot center cell
    if world[r, c] == 1:
        return True
    
    # Check a small circular region around the robot (simplified to square grid cells)
    # Convert radius to grid cells
    radius_cells = int(np.ceil(robot_radius_m / mparams.map_resolution))
    
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            # Check if within circular radius
            dist_cells = np.sqrt(dr**2 + dc**2)
            if dist_cells <= radius_cells:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if world[rr, cc] == 1:
                        return True
    
    return False


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

    # --- online policy for controls ---
    policy_cfg = cfg.get("policy", {})
    policy = make(policy_cfg["type"], **(policy_cfg.get("params", {}) or {})) if policy_cfg else None

    # --- reward function (optional) ---
    reward_cfg = cfg.get("reward", {})
    reward = make(reward_cfg["type"], **(reward_cfg.get("params", {}) or {})) if reward_cfg else None
    if reward is not None and hasattr(reward, "reset"):
        reward.reset()

    # --- start pose (meters, radians) ---
    sim = cfg.get("simulation", {})
    # random free-cell start and random yaw
    # Ensure we start away from walls (at least robot_radius away from borders)
    robot_radius_m = 0.15
    margin_cells = int(np.ceil(robot_radius_m / mparams.map_resolution)) + 1
    h, w = world.shape
    
    # Only consider cells that are free AND away from borders
    free = np.argwhere(world == 0)
    # Filter to keep only cells that are inside the safe margin
    safe_mask = (free[:, 0] >= margin_cells) & (free[:, 0] < h - margin_cells) & \
                (free[:, 1] >= margin_cells) & (free[:, 1] < w - margin_cells)
    safe_free = free[safe_mask]
    
    if len(safe_free) == 0:
        # Fallback: if no safe cells, use any free cell
        safe_free = free
        print("Warning: No safe cells found away from walls, using any free cell")
    
    r, c = safe_free[np.random.randint(len(safe_free))]
    x0, y0 = map_to_world(int(r), int(c), mparams)
    pose = (x0, y0, math.radians(360.0 * np.random.rand() - 180.0))

    # --- localization (EKF) --- initialize with correct starting pose
    planner_cfg = cfg.get("planner", {})
    if planner_cfg:
        planner_params = planner_cfg.get("params", {}) or {}
        loc = make(planner_cfg["type"], init_pose=pose, **(planner_params))
    else:
        loc = None

    # --- logs ---
    gt_path = [pose]
    ekf_path = [pose]
    # Per-step logs for learning supervision
    reward_total_log: list[float] = []
    coverage_log: list[float] = []

    # --- simulation loop ---
    steps = int(sim.get("steps", 400))
    cum_reward = 0.0
    for _ in range(steps):
        # sense at current GT pose
        ranges, angles_used = sensor.read(world, pose)

        # use current estimate for mapping before actuation
        est_pose = loc.get_pose() if (loc is not None and hasattr(loc, "get_pose")) else pose
        max_range_m = sensor.params.max_range_m if hasattr(sensor, "params") else 15.0
        mapper.update_with_scan(est_pose, ranges, angles_used, max_range_m)

        # get control from online policy
        if policy is not None and hasattr(policy, "next_control"):
            v, w = policy.next_control(ranges, angles_used)
        else:
            v, w = 0.0, 0.0

        # ground-truth evolve with collision detection
        new_pose = agent.step(pose, v, w)
        # Check for collision before accepting the new pose
        collision_detected = check_collision(world, new_pose, mparams)
        if collision_detected:
            # Collision detected - reject movement, keep current pose
            pose = pose  # Keep current pose
            # Don't apply controls that would cause collision
            v_actual, w_actual = 0.0, 0.0
        else:
            pose = new_pose
            v_actual, w_actual = v, w
        gt_path.append(pose)

        # Reward after mapping update and collision evaluation
        if reward is not None and hasattr(reward, "compute"):
            r_step, _ = reward.compute(mapper, world, pose, ranges, angles_used, collision_detected)
            cum_reward += float(r_step)
            reward_total_log.append(float(r_step))

        # Coverage (seen fraction) after mapping update
        if hasattr(mapper, "get_seen_mask"):
            coverage_log.append(float(mapper.get_seen_mask().mean()))
        else:
            coverage_log.append(float("nan"))

        # EKF predict using applied controls (only if no collision)
        if loc is not None and hasattr(loc, "predict"):
            loc.predict(v_actual, w_actual)
            est_pose = loc.get_pose() if hasattr(loc, "get_pose") else pose
            ekf_path.append(est_pose)

    # --- outputs ---
    out_dir = Path(__file__).parent / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # New folder for learning curves
    curves_dir = Path(__file__).parent / "results" / "learning_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    if 'reward' in cfg and reward is not None:
        print(f"Cumulative reward: {cum_reward:.3f}")
        if len(reward_total_log) > 0:
            plot_total_reward(reward_total_log, curves_dir)

    if len(coverage_log) > 0:
        plot_coverage(coverage_log, curves_dir)

    prob = mapper.to_prob()
    plot_occupancy_grid(prob, out_dir)

    if hasattr(mapper, "get_seen_mask"):
        seen_mask = mapper.get_seen_mask()
        plot_seen_mask(seen_mask, out_dir)

    plot_trajectories(
        world, gt_path, ekf_path if len(ekf_path) > 1 else None, mparams, out_dir
    )
    plot_combined(
        prob,
        world,
        gt_path,
        ekf_path if len(ekf_path) > 1 else None,
        mparams,
        out_dir,
    )


if __name__ == "__main__":
    run()