# SLAM Module Overview

This document summarizes the core SLAM (Simultaneous Localization and Mapping) components provided by this repository.

## Overview

The SLAM module provides the foundational algorithms and data structures for:

- **Localization**: Estimating robot pose in an unknown environment
- **Mapping**: Building a representation of the environment
- **Sensor Processing**: Handling LiDAR and other sensor data

## Key Components

### Mapping (`maps/`)

The mapping module provides:

- **MapParams**: Configuration for map dimensions and log-odds parameters
- **OccupancyGrid**: Log-odds based occupancy grid implementation
- **Map Utilities**:
  - `world_to_map()`: Convert world coordinates to grid indices
  - `map_to_world()`: Convert grid indices to world coordinates
  - `bresenham()`: Optimized ray tracing algorithm

**Occupancy Grid Algorithm**:

- Uses log-odds representation for numerical stability
- Updates cells along LiDAR rays using inverse sensor model
- Supports configurable log-odds thresholds for occupied/free space

### Localization (`planners/`)

**EKF Localization** (`ekf_localization.py`):

- Extended Kalman Filter for pose estimation
- Predicts pose using motion model (unicycle dynamics)
- Estimates state covariance for uncertainty quantification
- Supports configurable process and measurement noise

**State Representation**:

- Pose: `(x, y, θ)` in world coordinates
- Covariance: 3x3 matrix representing uncertainty

### Sensor Integration (`sensors/`)

**LiDAR Sensor** (`lidar.py`):

- Simulates LiDAR scans with configurable parameters
- Includes realistic noise models:
  - Range noise (Gaussian)
  - Angle noise (Gaussian)
  - Dropout (missing measurements)
- Supports field-of-view and beam configuration

### Map Generation (`maps/map.py`)

The map generation creates synthetic environments with:

- Border walls
- Rectangular obstacles
- Configurable resolution and size

## Usage Example

```python
from maps.map import MapParams, OccupancyGrid
from planners.ekf_localization import EKFLocalization

# Create map
params = MapParams(map_size=(200, 200), map_resolution=0.1)
grid = OccupancyGrid(params)

# Initialize localization
ekf = EKFLocalization(init_pose=(0.0, 0.0, 0.0))

# Update map with sensor data
pose = (0.0, 0.0, 0.0)
ranges = ...  # LiDAR ranges
angles = ...  # LiDAR angles (radians in sensor frame)
grid.update_with_scan(pose, ranges, angles, max_range_m=15.0)
```

Note: If you install this repository as a package named `SLAM`, you can alternatively import with the `SLAM.*` prefix (for example, `from SLAM.maps.map import MapParams`).

## Coordinate Systems

**World Coordinates**:

- Origin: Center or corner of map
- Units: Meters
- X: Right, Y: Up
- θ: Heading angle (radians)

**Map Coordinates**:

- Origin: Top-left of grid
- Units: Cell indices (row, col)
- Row: Increases downward
- Col: Increases rightward

## Performance Considerations

The mapping implementation includes optimizations:

- **Batch processing**: Vectorized updates for multiple cells
- **Generator-based ray tracing**: Memory-efficient Bresenham algorithm
- **Pre-computed trigonometric values**: Cached angle calculations

For large maps or high beam counts, consider:

- Reducing number of LiDAR beams
- Using lower map resolution
- Implementing spatial hashing for sparse maps

## Future Extensions

Potential enhancements:

- Loop closure detection
- Graph-based SLAM
- Particle filter localization
- Multi-robot SLAM
- Dynamic object handling
