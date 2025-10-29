from __future__ import annotations

from core.registry import register
from agents.unicycle import UnicycleAgent, UnicycleMotionConfig
from sensors.lidar import LidarSensor, LidarConfig
from mappers.occupancy_grid import OccupancyGridMapper
from SLAM.maps.map import MapParams


@register("agent.unicycle")
def _make_unicycle_agent(motion: dict | None = None):
    cfg = UnicycleMotionConfig(**(motion or {}))
    return UnicycleAgent(cfg)


@register("sensor.lidar")
def _make_lidar_sensor(params: dict | None = None, map_params: dict | None = None):
    p_cfg = LidarConfig(**(params or {}))
    m_cfg = MapParams(**(map_params or {})) if map_params else MapParams()
    return LidarSensor(p_cfg, m_cfg)


@register("mapper.occupancy_grid")
def _make_mapper(map_params: dict | None = None):
    m_cfg = MapParams(**(map_params or {})) if map_params else MapParams()
    return OccupancyGridMapper(m_cfg)


