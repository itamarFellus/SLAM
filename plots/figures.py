"""Legacy plotting module - kept for backward compatibility.

This module is deprecated. Please import directly from plots package:
    from plots import plot_occupancy_grid, plot_trajectories, ...

All plotting functions have been moved to organized modules:
- plots.learning_curves: Reward and coverage plots
- plots.map_plots: Occupancy grid and seen mask visualizations
- plots.trajectory_plots: Trajectory and combined visualizations
"""

# Re-export all plot functions for backward compatibility
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
