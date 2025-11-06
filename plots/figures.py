from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from maps.map import world_to_map

# Hard-coded base results directory
_BASE_RESULTS_DIR = Path(__file__).parent.parent / "results"
_PLOTS_DIR = _BASE_RESULTS_DIR / "plots"
_LEARNING_CURVES_DIR = _BASE_RESULTS_DIR / "learning_curves"


def plot_total_reward(reward_total_log: list[float]) -> None:
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


def plot_occupancy_grid(prob: np.ndarray) -> None:
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    plt.imshow(prob, origin="upper", cmap="gray")
    plt.title("Learned Occupancy Grid (probabilities)")
    plt.xlabel("x (cells)")
    plt.ylabel("y (cells)")
    plt.tight_layout()
    fig.savefig(_PLOTS_DIR / "occupancy_grid_final.png", dpi=150)
    plt.close(fig)


def plot_seen_mask(seen_mask: np.ndarray) -> None:
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    plt.imshow(seen_mask.astype(float), origin="upper", cmap="gray")
    plt.title("Seen Mask (1: seen, 0: unseen)")
    plt.xlabel("x (cells)")
    plt.ylabel("y (cells)")
    plt.tight_layout()
    fig.savefig(_PLOTS_DIR / "seen_mask.png", dpi=150)
    plt.close(fig)


def plot_trajectories(
    world: np.ndarray,
    gt_path: list[tuple[float, float, float]],
    ekf_path: Optional[list[tuple[float, float, float]]],
    mparams,
) -> None:
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    disp = np.ones_like(world, dtype=float)
    disp[world == 1] = 0.2

    xs_gt = [p[0] for p in gt_path]
    ys_gt = [p[1] for p in gt_path]
    pts_gt = np.array([world_to_map(x, y, mparams) for x, y in zip(xs_gt, ys_gt)])

    fig = plt.figure()
    plt.imshow(disp, origin="upper", cmap="gray")
    plt.plot(
        pts_gt[:, 1],
        pts_gt[:, 0],
        label="GT path",
        linewidth=2,
        color="red",
        linestyle="--",
        marker="o",
        markersize=2,
        alpha=1.0,
    )

    if ekf_path is not None and len(ekf_path) > 1:
        xs_ekf = [p[0] for p in ekf_path]
        ys_ekf = [p[1] for p in ekf_path]
        pts_ekf = np.array([world_to_map(x, y, mparams) for x, y in zip(xs_ekf, ys_ekf)])
        plt.plot(
            pts_ekf[:, 1],
            pts_ekf[:, 0],
            label="EKF path",
            linewidth=2,
            color="blue",
            marker="s",
            markersize=2,
            alpha=0.4,
        )

    plt.legend()
    plt.title("Trajectory over World (for sanity check)")
    plt.tight_layout()
    fig.savefig(_PLOTS_DIR / "trajectory_world.png", dpi=150)
    plt.close(fig)


def plot_combined(
    prob: np.ndarray,
    world: np.ndarray,
    gt_path: list[tuple[float, float, float]],
    ekf_path: Optional[list[tuple[float, float, float]]],
    mparams,
) -> None:
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    disp = np.ones_like(world, dtype=float)
    disp[world == 1] = 0.2

    xs_gt = [p[0] for p in gt_path]
    ys_gt = [p[1] for p in gt_path]
    pts_gt = np.array([world_to_map(x, y, mparams) for x, y in zip(xs_gt, ys_gt)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(disp, origin="upper", cmap="gray")
    ax1.plot(
        pts_gt[:, 1],
        pts_gt[:, 0],
        label="GT path",
        linewidth=2,
        color="red",
        linestyle="--",
        marker="o",
        markersize=3,
        alpha=1.0,
    )

    if ekf_path is not None and len(ekf_path) > 1:
        xs_ekf = [p[0] for p in ekf_path]
        ys_ekf = [p[1] for p in ekf_path]
        pts_ekf = np.array([world_to_map(x, y, mparams) for x, y in zip(xs_ekf, ys_ekf)])
        ax1.plot(
            pts_ekf[:, 1],
            pts_ekf[:, 0],
            label="EKF path",
            linewidth=2,
            color="blue",
            marker="s",
            markersize=3,
            alpha=0.4,
        )

    ax1.legend(fontsize=12)
    ax1.set_title("Trajectory over World", fontsize=16)
    ax1.set_xlabel("x (cells)", fontsize=12)
    ax1.set_ylabel("y (cells)", fontsize=12)

    ax2.imshow(prob, origin="upper", cmap="gray")
    ax2.set_title("Learned Occupancy Grid (probabilities)", fontsize=16)
    ax2.set_xlabel("x (cells)", fontsize=12)
    ax2.set_ylabel("y (cells)", fontsize=12)

    plt.tight_layout()
    fig.savefig(_PLOTS_DIR / "combined_trajectory_and_map.png", dpi=150)
    plt.close(fig)


