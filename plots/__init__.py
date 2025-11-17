"""Plotting utilities for robotics simulation.

All plotting functions are organized by category:
- learning_curves: Reward and coverage plots over time
- map_plots: Occupancy grid and seen mask visualizations
- trajectory_plots: Trajectory and combined visualizations
"""

from plots.learning_curves import plot_coverage, plot_total_reward
from plots.map_plots import plot_occupancy_grid, plot_seen_mask
from plots.trajectory_plots import plot_combined, plot_trajectories

__all__ = [
    "plot_coverage",
    "plot_total_reward",
    "plot_occupancy_grid",
    "plot_seen_mask",
    "plot_combined",
    "plot_trajectories",
]
