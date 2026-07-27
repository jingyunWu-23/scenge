"""Base utilities for straight-road LC test templates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

try:
    from envs.vehicle_state import VehicleState
except ImportError:  # pragma: no cover
    from carla_evolution.envs.vehicle_state import VehicleState

from tests.envs.scene_state import SceneState


@dataclass
class TemplateStepResult:
    scene: SceneState
    npc_actions: List[Dict[str, float]]


class StraightTemplate(ABC):
    name = "base"

    def __init__(self, route_length: float = 200.0, lane_width: float = 4.0, max_lanes: int = 3):
        self.route_length = float(route_length)
        self.lane_width = float(lane_width)
        self.max_lanes = int(max_lanes)

    @abstractmethod
    def reset(self, decoded_params: Dict[str, float], seed: int = 0) -> SceneState:
        raise NotImplementedError

    @abstractmethod
    def npc_intents(self, scene: SceneState, step: int, dt: float) -> List[Dict[str, float]]:
        raise NotImplementedError

    def static_lc_state(self, target_speed: float = 18.0, num_points: int = 40):
        x = np.linspace(0.0, self.route_length, int(num_points), dtype=np.float32)
        y = np.zeros_like(x, dtype=np.float32)
        return {"route": np.stack([x, y], axis=1), "target_speed": float(target_speed)}

    def _vehicle(
        self,
        role: str,
        index: int,
        route_s: float,
        speed: float,
        lane_id: int = 1,
    ) -> VehicleState:
        lane_id = int(np.clip(lane_id, 0, self.max_lanes - 1))
        route_s = float(route_s)
        return VehicleState(
            role=role,
            index=int(index),
            position=np.asarray([route_s, lane_id * self.lane_width], dtype=np.float32),
            speed=float(speed),
            lane_id=lane_id,
            raw_lane_id=lane_id,
            route_s=route_s,
            lateral_offset=0.0,
            heading=0.0,
            heading_error=0.0,
        )


def apply_intent(vehicle: VehicleState, intent: Dict[str, float], dt: float, max_speed: float = 35.0) -> None:
    if not getattr(vehicle, "active", True):
        return
    throttle = float(intent.get("throttle", 0.0))
    brake = float(intent.get("brake", 0.0))
    target_speed = intent.get("target_speed")
    if target_speed is not None:
        target_speed = float(target_speed)
        if vehicle.speed < target_speed:
            throttle = max(throttle, min((target_speed - vehicle.speed) / 4.0, 1.0))
            brake = 0.0
        else:
            brake = max(brake, min((vehicle.speed - target_speed) / 4.0, 1.0))
            throttle = 0.0
    accel = 2.5 * throttle - 4.0 * brake
    vehicle.speed = float(np.clip(vehicle.speed + accel * float(dt), 0.0, max_speed))
    lane_delta = int(intent.get("lane_delta", intent.get("lane_change", 0)))
    if lane_delta:
        target_lane = int(np.clip(int(vehicle.lane_id) + lane_delta, 0, 2))
        vehicle.set_lane(target_lane)
    distance = vehicle.speed * float(dt)
    vehicle.route_s = float((vehicle.route_s or 0.0) + distance)
    vehicle.position[0] = float(vehicle.route_s)
    vehicle.lateral_offset = 0.0
    vehicle.heading_error = 0.0


def detect_collision(vehicles: Iterable[VehicleState], min_gap: float = 4.8) -> bool:
    vehicle_list = [vehicle for vehicle in vehicles if getattr(vehicle, "physical_active", True)]
    for i, vehicle in enumerate(vehicle_list):
        for other in vehicle_list[i + 1 :]:
            if int(vehicle.lane_id) != int(other.lane_id):
                continue
            if abs(float((vehicle.route_s or vehicle.x) - (other.route_s or other.x))) <= float(min_gap):
                vehicle.crashed = True
                other.crashed = True
                return True
    return False
