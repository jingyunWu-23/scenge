"""Shared state object for straight-road LC tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    from envs.vehicle_state import VehicleState
except ImportError:  # pragma: no cover - allows direct package-style execution.
    from carla_evolution.envs.vehicle_state import VehicleState


@dataclass
class RouteState:
    route_id: str = "straight"
    route_length: float = 200.0
    ego_progress: float = 0.0
    waypoints: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=np.float32))

    def __post_init__(self) -> None:
        self.route_length = float(self.route_length)
        self.ego_progress = float(self.ego_progress)
        self.waypoints = np.asarray(self.waypoints, dtype=np.float32)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] < 2:
            raise ValueError(f"Route waypoints must have shape [N, >=2], got {self.waypoints.shape}.")


@dataclass
class LaneState:
    current_lane: int = 0
    left_lane_available: bool = False
    right_lane_available: bool = True
    lane_width: float = 4.0
    max_lanes: int = 4


@dataclass
class EventState:
    collision: bool = False
    timeout: bool = False
    off_route: bool = False
    cut_in_triggered: bool = False
    hard_brake_triggered: bool = False
    termination_reason: str = ""


@dataclass
class TemplateState:
    template_name: str = "straight_follow"
    decoded_params: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0


@dataclass
class SceneState:
    """Simulator-neutral scene state consumed by test observation adapters."""

    ego_vehicle: Optional[VehicleState] = None
    vehicles: List[VehicleState] = field(default_factory=list)
    route: RouteState = field(default_factory=RouteState)
    lanes: LaneState = field(default_factory=LaneState)
    events: EventState = field(default_factory=EventState)
    template: TemplateState = field(default_factory=TemplateState)

    @property
    def route_length(self) -> float:
        return self.route.route_length

    @property
    def route_completion(self) -> float:
        if self.route.route_length <= 1e-6:
            return 0.0
        return float(np.clip(self.route.ego_progress / self.route.route_length, 0.0, 1.0))

    @property
    def adv_cav_vehicles(self) -> List[VehicleState]:
        return [vehicle for vehicle in self.vehicles if vehicle.role in {"adv", "npc", "adversary"}]

    @property
    def hdv_vehicles(self) -> List[VehicleState]:
        return [vehicle for vehicle in self.vehicles if vehicle.role in {"hdv", "background"}]

    @classmethod
    def from_vehicle_states(
        cls,
        ego_vehicle: VehicleState,
        npc_vehicles: Optional[Iterable[VehicleState]] = None,
        route_length: float = 200.0,
        waypoints: Optional[Sequence[Sequence[float]]] = None,
        template_name: str = "straight_follow",
        decoded_params: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        lane_width: float = 4.0,
        max_lanes: int = 4,
    ) -> "SceneState":
        npc_list = list(npc_vehicles or [])
        vehicles = [ego_vehicle] + npc_list
        ego_progress = float(ego_vehicle.route_s if ego_vehicle.route_s is not None else ego_vehicle.x)
        if waypoints is None:
            waypoints = [[0.0, float(ego_vehicle.y)], [float(route_length), float(ego_vehicle.y)]]
        return cls(
            ego_vehicle=ego_vehicle,
            vehicles=vehicles,
            route=RouteState(route_length=route_length, ego_progress=ego_progress, waypoints=np.asarray(waypoints)),
            lanes=LaneState(
                current_lane=int(ego_vehicle.lane_id),
                left_lane_available=int(ego_vehicle.lane_id) > 0,
                right_lane_available=int(ego_vehicle.lane_id) < int(max_lanes) - 1,
                lane_width=lane_width,
                max_lanes=max_lanes,
            ),
            template=TemplateState(template_name=template_name, decoded_params=decoded_params or {}, seed=int(seed)),
        )

    def sync_route_progress_from_ego(self) -> None:
        if self.ego_vehicle is None:
            self.route.ego_progress = 0.0
            return
        self.route.ego_progress = float(
            self.ego_vehicle.route_s if self.ego_vehicle.route_s is not None else self.ego_vehicle.x
        )
