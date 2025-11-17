# %%
# main.py
import math
from pathlib import Path

import numpy as np
import yaml

from maps.map import MapParams, make_world_grid, world_to_map, map_to_world, OccupancyGrid
from agent.lidar_based import LidarParams, MotionParams, LidarAgent, simulate_lidar_scan
from rewards.exploration_reward import ExplorationReward, ExplorationRewardConfig
from planners.ekf_localization import EKFLocalization, EKFInit
from policies.reward_based_explorer import RewardBasedExplorer, RewardBasedConfig
from plots import (
    plot_combined,
    plot_seen_mask,
    plot_total_reward,
    plot_trajectories,
)


def run() -> None:
    # --- load configuration ---
    cfg_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # --- params (from config) ---
    map_cfg = cfg.get("map", {}) or {}
    mparams = MapParams(**map_cfg)

    agent_motion_cfg = ((cfg.get("agent") or {}).get("params") or {}).get("motion") or {}
    mot = MotionParams(**agent_motion_cfg)

    sensor_section = (cfg.get("sensor") or {}).get("params") or {}
    lidar_params_cfg = (sensor_section.get("params") or {})
    lparams = LidarParams(**lidar_params_cfg)

    # --- world & mapper ---
    world = make_world_grid(mparams)
    og = OccupancyGrid(mparams)

    # --- agent ---
    agent = LidarAgent(lparams, mot)

    # --- reward function ---
    # Prefer top-level reward config if present; else fall back to policy's reward config
    reward_top_cfg = (((cfg.get("reward") or {}).get("params") or {}).get("config") or {})
    policy_section = (cfg.get("policy") or {}).get("params") or {}
    policy_reward_cfg = ((policy_section.get("reward") or {}).get("config") or {})
    reward_config_use = reward_top_cfg if len(reward_top_cfg) > 0 else policy_reward_cfg
    reward_fn = ExplorationReward(config=ExplorationRewardConfig(**(reward_config_use or {})))
    reward_fn.reset()

    # --- start pose (meters, radians) ---
    # Generate random starting position strictly inside the map bounds,
    # on a free cell, AND at least 0.5 meters away from any obstacle/border.
    h, w = mparams.map_size

    # Clearance requirement in cells
    clearance_m = 0.5
    clearance_cells = int(math.ceil(clearance_m / mparams.map_resolution))
    rr2 = clearance_cells * clearance_cells

    def has_clearance(world_grid: np.ndarray, r: int, c: int) -> bool:
        """Return True if (r,c) is at least clearance_cells away from any obstacle or map boundary.
        Uses Euclidean distance on the cell grid.
        """
        h_loc, w_loc = world_grid.shape
        # The center cell must be free
        if r < 0 or r >= h_loc or c < 0 or c >= w_loc or world_grid[r, c] == 1:
            return False
        for dr in range(-clearance_cells, clearance_cells + 1):
            rr = r + dr
            for dc in range(-clearance_cells, clearance_cells + 1):
                # Inside circle of radius clearance_cells
                if dr * dr + dc * dc > rr2:
                    continue
                cc = c + dc
                # Treat out-of-bounds as obstacle to enforce margin from borders
                if rr < 0 or rr >= h_loc or cc < 0 or cc >= w_loc:
                    return False
                if world_grid[rr, cc] == 1:
                    return False
        return True

    found_free = False
    for _ in range(1000):
        start_row = np.random.randint(1, h - 1)
        start_col = np.random.randint(1, w - 1)
        if has_clearance(world, start_row, start_col):
            found_free = True
            break
    if not found_free:
        # Fallback: build a list of all cells that satisfy clearance and sample
        candidates = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if has_clearance(world, r, c):
                    candidates.append((r, c))
        if len(candidates) == 0:
            # As a last resort (should not happen with current maps), use center
            start_row, start_col = h // 2, w // 2
        else:
            start_row, start_col = candidates[np.random.randint(len(candidates))]
    x0, y0 = map_to_world(start_row, start_col, mparams)
    # Random orientation in [0, 2π)
    start_yaw = np.random.uniform(0, 2 * math.pi)
    pose = (x0, y0, start_yaw)

    # --- EKF & policy ---
    ekf_init_cfg = ((cfg.get("planner") or {}).get("params") or {}).get("init") or {}
    ekf = EKFLocalization(init_pose=pose, cfg=EKFInit(**ekf_init_cfg))

    policy_cfg_vals = policy_section.get("config") or {}
    policy = RewardBasedExplorer(
        ekf=ekf,
        lidar_params=lparams,
        map_params=mparams,
        reward_cfg=ExplorationRewardConfig(**(policy_reward_cfg or {})),
        cfg=RewardBasedConfig(**policy_cfg_vals),
    )

    # --- logs ---
    gt_path = [pose]
    ekf_path = [ekf.get_pose()]
    reward_log = []

    # --- simulation loop ---
    steps = int(((cfg.get("simulation") or {}).get("steps")) or 400)
    collision_rc = None
    for k in range(steps):
        # simulate lidar at current position
        scan_pose = pose
        ranges, angles_used = simulate_lidar_scan(world, scan_pose, lparams, mparams)

        # policy decides next control based on sensor data and expected reward
        v, w = policy.next_control(ranges, angles_used, og, world, scan_pose)

        # ground-truth evolve
        pose = agent.step(pose, v, w)
        gt_path.append(pose)
        # record EKF estimate advanced by the policy
        ekf_path.append(ekf.get_pose())

        # mapping update using the scan from the previous position
        og.update_with_scan(scan_pose, ranges, angles_used, lparams.max_range_m)

        # Check for collision (agent position in occupied cell)
        row, col = world_to_map(pose[0], pose[1], mparams)
        h, w_map = world.shape
        collision_detected = (
            row < 0 or row >= h or col < 0 or col >= w_map or world[row, col] == 1
        )

        # Compute reward
        reward, reward_info = reward_fn.compute(
            og, world, pose, ranges, angles_used, collision_detected
        )
        reward_log.append(reward)

        # If collided: clamp to map bounds for plotting, record and stop
        if collision_detected:
            r_clamp = int(max(0, min(h - 1, row)))
            c_clamp = int(max(0, min(w_map - 1, col)))
            collision_rc = (r_clamp, c_clamp)
            break

    # --- outputs ---
    # Sanity-plot trajectory over the world
    plot_trajectories(world, gt_path, ekf_path, mparams)

    prob = og.to_prob()
    plot_combined(prob, world, gt_path, ekf_path, mparams, collision_rc=collision_rc)
    
    # Plot seen mask
    seen_mask = og.get_seen_mask()
    plot_seen_mask(seen_mask)
    
    # Plot reward
    plot_total_reward(reward_log)


if __name__ == "__main__":
    run()
# %%
