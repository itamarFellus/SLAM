"""Learning curve plotting functions."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Hard-coded base results directory
_BASE_RESULTS_DIR = Path(__file__).parent.parent / "results"
_LEARNING_CURVES_DIR = _BASE_RESULTS_DIR / "learning_curves"


def plot_total_reward(reward_total_log: list[float]) -> None:
    """Plot total reward over time."""
    _LEARNING_CURVES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    plt.plot(np.arange(len(reward_total_log)), reward_total_log, color="C0")
    plt.title("Total Reward vs Steps")
    plt.xlabel("Step")
    plt.ylabel("Reward (per step)")
    plt.tight_layout()
    fig.savefig(_LEARNING_CURVES_DIR / "total_reward.png", dpi=150)
    plt.close(fig)


def plot_coverage(coverage_log: list[float]) -> None:
    """Plot coverage (seen fraction) over time."""
    _LEARNING_CURVES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    plt.plot(np.arange(len(coverage_log)), coverage_log, color="C2")
    plt.title("Coverage vs Steps")
    plt.xlabel("Step")
    plt.ylabel("Seen fraction")
    plt.ylim(0, 1)
    plt.tight_layout()
    fig.savefig(_LEARNING_CURVES_DIR / "coverage.png", dpi=150)
    plt.close(fig)

