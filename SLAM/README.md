# SLAM - Simultaneous Localization and Mapping Simulation

A modular SLAM (Simultaneous Localization and Mapping) simulation framework implemented in Python. This project demonstrates EKF-based localization combined with occupancy grid mapping using LiDAR sensor data.

## Features

- **Modular Architecture**: Registry-based component system for easy extensibility
- **EKF Localization**: Extended Kalman Filter for robot pose estimation
- **Occupancy Grid Mapping**: Log-odds based occupancy grid mapping
- **LiDAR Simulation**: Configurable LiDAR sensor with noise and dropout models
- **Unicycle Robot Model**: Realistic unicycle motion model for mobile robots
- **Reactive Exploration Policy**: Autonomous exploration with obstacle avoidance
- **Collision Detection**: Real-time collision checking with robot radius consideration
- **Visualization**: Comprehensive trajectory and map visualization

## Project Structure

```
SLAM/
├── agents/              # Robot agent implementations
│   └── unicycle.py     # Unicycle motion model
├── core/               # Core system components
│   └── registry.py    # Component registry pattern
├── configs/            # Configuration files
│   └── default.yaml   # Default simulation configuration
├── interfaces/         # Abstract base classes
├── maps/              # Map generation and utilities
│   └── map.py         # Map parameters and occupancy grid
├── mappers/           # Mapping algorithms
│   └── occupancy_grid.py
├── planners/          # Localization planners
│   └── ekf_localization.py
├── policies/          # Control policies
│   └── reactive_explorer.py
├── sensors/           # Sensor implementations
│   └── lidar.py       # LiDAR sensor simulation
├── results/           # Output directory for plots
├── main.py            # Main simulation script
└── plugins.py         # Component registration

```

## Installation

### Prerequisites

- Python 3.8+
- NumPy
- Matplotlib
- PyYAML

### Setup

1. Clone or navigate to the project directory
2. Install required packages:

```bash
pip install numpy matplotlib pyyaml
```

## Usage

### Basic Usage

Run the simulation with default configuration:

```bash
python main.py
```

### Configuration

Edit `configs/default.yaml` to customize:

- **Map parameters**: Size, resolution, log-odds thresholds
- **Agent settings**: Motion model parameters (velocity, angular velocity, timestep)
- **Sensor parameters**: LiDAR FOV, number of beams, range, noise models
- **Localization**: EKF initialization parameters
- **Policy**: Exploration behavior parameters
- **Simulation**: Number of steps

Example configuration:

```yaml
map:
  map_size: [200, 200]
  map_resolution: 0.1
  lo_occ: 0.85
  lo_free: 0.4

agent:
  type: agent.unicycle
  params:
    motion:
      v_nom: 0.4
      w_nom: 0.3
      step_dt: 0.2

sensor:
  type: sensor.lidar
  params:
    params:
      fov_deg: 270.0
      num_beams: 180
      max_range_m: 15.0
      sigma_r: 0.02
      dropout_prob: 0.02

simulation:
  steps: 600
```

## Output

The simulation generates three visualization plots in `results/plots/`:

1. **occupancy_grid_final.png**: Learned occupancy grid probabilities
2. **trajectory_world.png**: Ground-truth and EKF-estimated trajectories overlaid on the world map
3. **combined_trajectory_and_map.png**: Side-by-side comparison of trajectories and learned map

### Trajectory Visualization

- **Red dashed line**: Ground-truth (GT) path with circular markers
- **Blue solid line**: EKF-estimated path with square markers
- Paths are semi-transparent to avoid obscuring the background

## Architecture

### Registry System

The project uses a registry pattern for component management:

```python
from core.registry import register, make

@register("component.name")
def factory_function(params):
    return Component(params)

# Later...
component = make("component.name", **kwargs)
```

### Components

**Agents** (`agents/`): Robot motion models

- `UnicycleAgent`: Unicycle differential drive model

**Sensors** (`sensors/`): Sensor simulators

- `LidarSensor`: LiDAR with configurable FOV, noise, and dropout

**Mappers** (`mappers/`): Mapping algorithms

- `OccupancyGridMapper`: Log-odds occupancy grid mapping with optimized Bresenham ray tracing

**Planners** (`planners/`): Localization algorithms

- `EKFLocalization`: Extended Kalman Filter for pose estimation

**Policies** (`policies/`): Control policies

- `ReactiveExplorer`: Reactive exploration with obstacle avoidance

## Algorithm Details

### EKF Localization

The Extended Kalman Filter estimates robot pose (x, y, θ) using:

- Motion model: Unicycle dynamics
- Prediction step: State propagation with control inputs
- Update step: Sensor measurements (currently uses pose estimates for mapping)

### Occupancy Grid Mapping

- **Log-odds representation**: Uses log-odds for numerical stability
- **Ray tracing**: Optimized Bresenham's algorithm with batch processing
- **Update rules**:
  - Free space: Decrement log-odds along rays
  - Occupied space: Increment log-odds at hit points
  - Clamping: Bounds log-odds to prevent overflow

### Collision Detection

- Checks robot radius against obstacles
- Prevents out-of-bounds movement
- Rejects movements that would cause collisions

## Performance Optimizations

- **Vectorized operations**: NumPy batch processing for map updates
- **Generator-based ray tracing**: Memory-efficient Bresenham implementation
- **Pre-computed trigonometric values**: Cached cos/sin for angle calculations
- **Bounds filtering**: Efficient numpy masking for valid cells

## Extending the System

### Adding a New Component

1. Implement the component class following the interface in `interfaces/`
2. Register it in `plugins.py`:

```python
@register("component.name")
def _make_component(params: dict | None = None):
    return Component(**params)
```

3. Add configuration in `configs/default.yaml`

### Example: Adding a New Sensor

```python
# In plugins.py
@register("sensor.custom")
def _make_custom_sensor(params: dict | None = None):
    cfg = CustomSensorConfig(**(params or {}))
    return CustomSensor(cfg)
```

## Configuration Parameters

### Map Parameters

- `map_size`: Grid dimensions `[rows, cols]`
- `map_resolution`: Meters per cell
- `lo_occ`: Log-odds increment for occupied cells
- `lo_free`: Log-odds decrement for free cells
- `lo_min/max`: Log-odds bounds

### Agent Parameters

- `v_nom`: Nominal linear velocity (m/s)
- `w_nom`: Nominal angular velocity (rad/s)
- `step_dt`: Time step (seconds)

### Sensor Parameters

- `fov_deg`: Field of view in degrees
- `num_beams`: Number of LiDAR beams
- `max_range_m`: Maximum detection range
- `sigma_r`: Range noise standard deviation
- `dropout_prob`: Probability of measurement dropout
- `angle_noise_deg`: Angle noise in degrees

### Policy Parameters

- `v_fwd`: Forward velocity
- `w_gain`: Angular velocity gain
- `min_clearance_m`: Minimum obstacle clearance

## License

This project is intended for educational and research purposes.

## Contributing

Feel free to extend the system with:

- New sensor models
- Different localization algorithms (particle filter, graph-based SLAM)
- Alternative mapping approaches (feature-based, graph-based)
- New exploration policies

## References

- Probabilistic Robotics (Thrun, Burgard, Fox)
- SLAM literature on EKF and occupancy grid mapping
