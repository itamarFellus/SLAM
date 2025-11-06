from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np

from interfaces.reward import Reward


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    # Numerical stability: clip away from 0 and 1
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


@dataclass
class ExplorationRewardConfig:
    # Weights
    weight_unknown_reduction: float = 1.0
    weight_uncertainty_reduction: float = 0.2
    weight_collision_risk: float = 0.8
    collision_penalty: float = 3.0
    step_penalty: float = 0.01

    # Safety
    min_clearance_m: float = 0.6

    # Normalization
    normalize_by_total_cells: bool = True


class ExplorationReward(Reward):
    def __init__(self, config: ExplorationRewardConfig | None = None) -> None:
        self.cfg = config or ExplorationRewardConfig()
        self._prev_seen: np.ndarray | None = None
        self._prev_log_odds: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_seen = None
        self._prev_log_odds = None

    def _ensure_prev(self, mapper: any) -> None:
        # Lazily initialize previous state on first call
        if self._prev_seen is None:
            # Treat start as completely unknown/unseen
            prob = mapper.to_prob()
            self._prev_seen = np.zeros_like(prob, dtype=bool)
        if self._prev_log_odds is None:
            # Start from neutral log-odds (0 => p=0.5)
            if hasattr(mapper, "to_log_odds"):
                shape = mapper.to_log_odds().shape
                self._prev_log_odds = np.zeros(shape, dtype=np.float32)
            else:
                prob = mapper.to_prob()
                self._prev_log_odds = np.zeros_like(prob, dtype=np.float32)

    def compute(
        self,
        mapper: any,
        world: np.ndarray,
        pose: Tuple[float, float, float],
        ranges: np.ndarray,
        angles: np.ndarray,
        collision_detected: bool,
    ) -> Tuple[float, Dict[str, float]]:
        self._ensure_prev(mapper)

        # Current map state
        seen_now = mapper.get_seen_mask() if hasattr(mapper, "get_seen_mask") else None
        prob_now = mapper.to_prob()

        # Unknown reduction: newly seen cells since previous step
        if seen_now is not None and self._prev_seen is not None:
            newly_seen = np.logical_and(seen_now, np.logical_not(self._prev_seen))
            unknown_reduction = float(np.count_nonzero(newly_seen))
            denom_unk = float(np.prod(seen_now.shape)) if self.cfg.normalize_by_total_cells else 1.0
            unknown_term = (unknown_reduction / denom_unk) * self.cfg.weight_unknown_reduction
        else:
            unknown_term = 0.0

        # Uncertainty reduction: global binary entropy reduction
        # Compute from prev and current probabilities
        prev_lo = self._prev_log_odds
        prev_p = 1.0 - 1.0 / (1.0 + np.exp(prev_lo))
        ent_prev = _binary_entropy(prev_p)
        ent_now = _binary_entropy(prob_now)
        # Only count reductions
        ent_delta = ent_prev - ent_now
        ent_delta[ent_delta < 0.0] = 0.0
        total_ent_reduction = float(np.sum(ent_delta))
        denom_ent = float(np.prod(prob_now.shape)) if self.cfg.normalize_by_total_cells else 1.0
        uncertainty_term = (total_ent_reduction / denom_ent) * self.cfg.weight_uncertainty_reduction

        # Collision risk penalty from min range
        if ranges.size > 0:
            r_min = float(np.min(ranges))
            clearance_violation = max(0.0, self.cfg.min_clearance_m - r_min)
            risk = clearance_violation / max(self.cfg.min_clearance_m, 1e-6)
        else:
            risk = 0.0
        collision_risk_penalty = -self.cfg.weight_collision_risk * risk

        # Actual collision penalty
        collision_penalty = -self.cfg.collision_penalty if collision_detected else 0.0

        # Step penalty
        step_penalty = -self.cfg.step_penalty

        reward = unknown_term + uncertainty_term + collision_risk_penalty + collision_penalty + step_penalty

        # Update previous state for next step
        if seen_now is not None:
            self._prev_seen = seen_now.copy()
        # Store current log-odds as previous for next call
        if hasattr(mapper, "to_log_odds"):
            self._prev_log_odds = mapper.to_log_odds()
        else:
            p_now = np.clip(prob_now, 1e-6, 1 - 1e-6)
            self._prev_log_odds = np.log(p_now / (1 - p_now)).astype(np.float32)

        info = {
            "unknown_term": float(unknown_term),
            "uncertainty_term": float(uncertainty_term),
            "collision_risk_penalty": float(collision_risk_penalty),
            "collision_penalty": float(collision_penalty),
            "step_penalty": float(-self.cfg.step_penalty),
            "total": float(reward),
        }
        return float(reward), info


