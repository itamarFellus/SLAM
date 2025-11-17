from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import copy
import numpy as np

from planners.ekf_localization import EKFLocalization
from rewards.exploration_reward import ExplorationReward, ExplorationRewardConfig
from maps.map import MapParams, world_to_map, OccupancyGrid, bresenham
from agent.lidar_based import simulate_lidar_scan, LidarParams


@dataclass
class RewardBasedConfig:
    """Configuration for the reward-guided exploration policy."""
    v_fwd: float = 0.4
    w_max: float = 0.6
    num_w_candidates: int = 7
    lookahead_steps: int = 1  # single-step greedy by default
    exploration_noise: float = 0.0  # optional stochasticity


class RewardBasedExplorer:
    """
    Greedy, reward-guided policy:
    - Enumerates angular velocity candidates (fixed forward speed)
    - Predicts next pose using an EKF copy (no side effects)
    - Simulates a LiDAR scan at the predicted pose
    - Applies a hypothetical mapping update on a mapper copy
    - Scores the candidate with a fresh ExplorationReward instance whose
      "previous state" is the current map, so reward reflects the delta
    - Selects the action with the highest expected reward
    """
    def __init__(
        self,
        ekf: EKFLocalization,
        lidar_params: LidarParams,
        map_params: MapParams,
        reward_cfg: ExplorationRewardConfig | None = None,
        cfg: RewardBasedConfig | None = None,
    ) -> None:
        self.ekf = ekf
        self.lidar_params = lidar_params
        self.map_params = map_params
        self.cfg = cfg or RewardBasedConfig()
        self.reward_cfg = reward_cfg or ExplorationRewardConfig()

    def _enumerate_candidates(self) -> List[Tuple[float, float]]:
        """Generate (v, w) candidates."""
        if self.cfg.num_w_candidates <= 1:
            ws = [0.0]
        else:
            ws = list(np.linspace(-self.cfg.w_max, self.cfg.w_max, self.cfg.num_w_candidates))
        v = self.cfg.v_fwd
        return [(v, w) for w in ws]

    def _predict_pose(self, pose: Tuple[float, float, float], v: float, w: float) -> Tuple[float, float, float]:
        """Predict a single-step pose using a copy of the EKF (keeps EKF in this policy side-effect free)."""
        ekf_copy = copy.deepcopy(self.ekf)
        ekf_copy.x = np.array(pose, dtype=float).reshape(3, 1)
        ekf_copy.predict(v, w)
        return ekf_copy.get_pose()

    def _will_collide(self, world: np.ndarray, pose: Tuple[float, float, float]) -> bool:
        r, c = world_to_map(pose[0], pose[1], self.map_params)
        h, w_map = world.shape
        if r < 0 or r >= h or c < 0 or c >= w_map:
            return True
        return world[r, c] == 1

    def _copy_mapper(self, mapper: OccupancyGrid) -> OccupancyGrid:
        """Deep copy mapper arrays for hypothetical updates."""
        m = OccupancyGrid(self.map_params)
        m.log_odds = mapper.log_odds.copy()
        m.seen = mapper.seen.copy()
        return m

    def _path_collides(self, world: np.ndarray, start_pose: Tuple[float, float, float], end_pose: Tuple[float, float, float]) -> bool:
        """Check for collisions along the straight-line path between poses using grid traversal.

        Uses Bresenham over map indices from start to end; any occupied or out-of-bounds cell implies collision.
        """
        r0, c0 = world_to_map(start_pose[0], start_pose[1], self.map_params)
        r1, c1 = world_to_map(end_pose[0], end_pose[1], self.map_params)
        h, w_map = world.shape
        for r, c in bresenham(r0, c0, r1, c1):
            if r < 0 or r >= h or c < 0 or c >= w_map:
                return True
            if world[r, c] == 1:
                return True
        return False

    def next_control(
        self,
        ranges: np.ndarray,
        angles_used: np.ndarray,
        mapper: OccupancyGrid,
        world: np.ndarray,
        current_pose: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        candidates = self._enumerate_candidates()

        best_reward = -np.inf
        best_action: Tuple[float, float] | None = None

        # Snapshot of "previous" state for reward delta
        prev_seen = mapper.get_seen_mask() if hasattr(mapper, "get_seen_mask") else None
        prev_log_odds = mapper.log_odds.copy()

        for v, w in candidates:
            # Predict pose
            pred_pose = self._predict_pose(current_pose, v, w)

            # Hard-skip any candidate whose swept path intersects obstacles or bounds
            if self._path_collides(world, current_pose, pred_pose):
                continue

            # Simulate future scan
            cand_ranges, cand_angles = simulate_lidar_scan(world, pred_pose, self.lidar_params, self.map_params)

            # Hypothetical mapping update on a mapper copy
            mapper_copy = self._copy_mapper(mapper)
            mapper_copy.update_with_scan(pred_pose, cand_ranges, cand_angles, self.lidar_params.max_range_m)

            # Collision check at predicted pose
            collision = self._will_collide(world, pred_pose)

            # Fresh reward instance with "previous" set to current map
            rwd = ExplorationReward(config=self.reward_cfg)
            # Set previous buffers so reward reflects improvement from current -> candidate
            if prev_seen is not None:
                rwd._prev_seen = prev_seen.copy()
            rwd._prev_log_odds = prev_log_odds.copy()

            # Score
            cand_reward, _ = rwd.compute(mapper_copy, world, pred_pose, cand_ranges, cand_angles, collision)

            # Optional small exploration noise
            if self.cfg.exploration_noise > 0.0:
                cand_reward += np.random.normal(scale=self.cfg.exploration_noise)

            if cand_reward > best_reward:
                best_reward = cand_reward
                best_action = (v, w)

        if best_action is None:
            return 0.0, 0.0

        # Keep EKF in sync with chosen action (policy's internal estimate)
        self.ekf.predict(best_action[0], best_action[1])
        return best_action


