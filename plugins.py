from __future__ import annotations

from core.registry import register
from agents.unicycle import UnicycleAgent, UnicycleMotionConfig
from sensors.lidar import LidarSensor, LidarConfig
from mappers.occupancy_grid import OccupancyGridMapper
from maps.map import MapParams
from planners.ekf_localization import EKFLocalization, EKFInit
from policies.reactive_explorer import ReactiveExplorer, ExplorerConfig
from rewards.exploration_reward import ExplorationReward, ExplorationRewardConfig


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



@register("planner.ekf_localization")
def _make_ekf_localization(init: dict | None = None, init_pose: tuple | None = None):
    init_pose_tuple = init_pose if init_pose is not None else (0.0, 0.0, 0.0)
    return EKFLocalization(init_pose=init_pose_tuple, cfg=EKFInit(**(init or {})))


@register("policy.reactive_explorer")
def _make_reactive_explorer(config: dict | None = None):
    return ReactiveExplorer(cfg=ExplorerConfig(**(config or {})))



@register("reward.exploration")
def _make_exploration_reward(config: dict | None = None):
    return ExplorationReward(config=ExplorationRewardConfig(**(config or {})))

