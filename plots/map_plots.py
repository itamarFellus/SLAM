"""Map visualization plotting functions."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Hard-coded base results directory
_BASE_RESULTS_DIR = Path(__file__).parent.parent / "results"
_PLOTS_DIR = _BASE_RESULTS_DIR / "plots"


def plot_occupancy_grid(prob: np.ndarray) -> None:
    """Plot the learned occupancy grid as probabilities."""
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
    """Plot the seen mask showing which cells have been observed."""
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    plt.imshow(seen_mask.astype(float), origin="upper", cmap="gray")
    plt.title("Seen Mask (1: seen, 0: unseen)")
    plt.xlabel("x (cells)")
    plt.ylabel("y (cells)")
    plt.tight_layout()
    fig.savefig(_PLOTS_DIR / "seen_mask.png", dpi=150)
    plt.close(fig)

