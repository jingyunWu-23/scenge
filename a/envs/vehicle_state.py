"""Shared vehicle state used by observations, rewards, and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class StraightLane:
    """Minimal lane object compatible with the old reward code."""

    def __init__(self, lane_id: int, width: float = 4.0):
        self.lane_id = int(lane_id)
        self.width = float(width)

    def local_coordinates(self, position) -> Tuple[float, float]:
        position = np.asarray(position, dtype=float)
        center_y = self.lane_id * self.width
        return float(position[0]), float(position[1] - center_y)


@dataclass
class VehicleState:
    """Simulator-neutral vehicle state.

    The attributes intentionally match the subset used by the existing PPO,
    reward, and wrapper code.
    """

    role: str
    index: int
    position: np.ndarray
    speed: float
    lane_id: int = 0
    heading: float = 0.0
    crashed: bool = False
    on_road: bool = True
    active: bool = True
    physical_active: bool = True
    route_s: Optional[float] = None
    lateral_offset: float = 0.0
    heading_error: float = 0.0
    road_id: int = 0
    section_id: int = 0
    raw_lane_id: int = 0
    obstacle_distance: float = -1.0
    obstacle_lateral: float = 0.0
    obstacle_risk: float = 0.0
    target_lane_obstacle_risk: float = 0.0
    lane_obstacle_veto: bool = False
    lane_change_cancelled_for_obstacle: bool = False
    hazardous_lane_change: bool = False
    hazardous_lane_change_risk: float = 0.0
    obstacle_takeover_active: bool = False
    obstacle_takeover_released: bool = False
    length: float = 4.8
    width: float = 2.0

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)
        self.speed = float(self.speed)
        self.lane_id = int(self.lane_id)
        self.route_s = None if self.route_s is None else float(self.route_s)
        self.lateral_offset = float(self.lateral_offset)
        self.heading_error = float(self.heading_error)
        self.road_id = int(self.road_id)
        self.section_id = int(self.section_id)
        self.raw_lane_id = int(self.raw_lane_id if self.raw_lane_id != 0 else self.lane_id)
        self.obstacle_distance = float(self.obstacle_distance)
        self.obstacle_lateral = float(self.obstacle_lateral)
        self.obstacle_risk = float(self.obstacle_risk)
        self.target_lane_obstacle_risk = float(self.target_lane_obstacle_risk)
        self.lane_obstacle_veto = bool(self.lane_obstacle_veto)
        self.lane_change_cancelled_for_obstacle = bool(self.lane_change_cancelled_for_obstacle)
        self.hazardous_lane_change = bool(self.hazardous_lane_change)
        self.hazardous_lane_change_risk = float(self.hazardous_lane_change_risk)
        self.obstacle_takeover_active = bool(self.obstacle_takeover_active)
        self.obstacle_takeover_released = bool(self.obstacle_takeover_released)
        self.lane = StraightLane(self.lane_id)
        self.lane_index = (self.road_id, self.section_id, self.raw_lane_id)

    def set_lane(self, lane_id: int):
        self.lane_id = int(lane_id)
        self.raw_lane_id = int(lane_id)
        self.lane = StraightLane(self.lane_id)
        self.lane_index = (self.road_id, self.section_id, self.raw_lane_id)
        self.position[1] = float(self.lane_id) * self.lane.width

    def lane_distance_to(self, other: "VehicleState") -> float:
        return float(other.position[0] - self.position[0] - 0.5 * (self.length + other.length))

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

def heading_forward_vector(vehicle: VehicleState) -> np.ndarray:
    heading = float(getattr(vehicle, "heading", 0.0))
    return np.asarray([np.cos(heading), np.sin(heading)], dtype=float)


def same_lane_segment(vehicle: VehicleState, other: VehicleState) -> bool:
    return bool(
        int(vehicle.road_id) == int(other.road_id)
        and int(vehicle.section_id) == int(other.section_id)
        and int(vehicle.raw_lane_id) == int(other.raw_lane_id)
    )


def longitudinal_gap(vehicle: VehicleState, other: VehicleState) -> float:
    vehicle_s = getattr(vehicle, "route_s", None)
    other_s = getattr(other, "route_s", None)
    if vehicle_s is not None and other_s is not None:
        return float(other_s - vehicle_s)
    origin = np.asarray(vehicle.position, dtype=float)
    target = np.asarray(other.position, dtype=float)
    return float(np.dot(target - origin, heading_forward_vector(vehicle)))
