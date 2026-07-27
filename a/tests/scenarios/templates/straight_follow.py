"""Straight-follow LC templates: sudden deceleration, speed oscillation, hard brake."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from tests.envs.scene_state import SceneState
from .base import StraightTemplate


FOLLOW_BEHAVIORS = ("sudden_decel", "speed_oscillation", "hard_brake")


class StraightFollowTemplate(StraightTemplate):
    name = "straight_follow"

    def reset(self, decoded_params: Dict[str, float], seed: int = 0) -> SceneState:
        ego_lane = int(decoded_params.get("ego_lane", 1))
        ego_speed = float(decoded_params.get("ego_speed", 18.0))
        front_offset = float(decoded_params.get("front_offset", 30.0))
        front_initial_speed = float(decoded_params.get("front_initial_speed", 14.0))

        ego = self._vehicle("ego", 0, route_s=0.0, speed=ego_speed, lane_id=ego_lane)
        front = self._vehicle(
            "npc",
            0,
            route_s=max(front_offset, 6.0),
            speed=front_initial_speed,
            lane_id=ego_lane,
        )
        return SceneState.from_vehicle_states(
            ego,
            [front],
            route_length=self.route_length,
            template_name=self.name,
            decoded_params=dict(decoded_params),
            seed=int(seed),
            lane_width=self.lane_width,
            max_lanes=self.max_lanes,
        )

    def npc_intents(self, scene: SceneState, step: int, dt: float) -> List[Dict[str, float]]:
        params = scene.template.decoded_params
        behavior = str(params.get("behavior_type", "sudden_decel"))
        trigger_distance = float(params.get("trigger_distance", 18.0))
        front = scene.adv_cav_vehicles[0] if scene.adv_cav_vehicles else None
        ego = scene.ego_vehicle
        if front is None or ego is None:
            return []

        gap = float((front.route_s or front.x) - (ego.route_s or ego.x))
        triggered = gap <= trigger_distance or bool(params.get("_triggered", False))
        params["_triggered"] = triggered

        if not triggered:
            return [{"target_speed": float(params.get("front_initial_speed", front.speed))}]

        if behavior == "speed_oscillation":
            base_speed = float(params.get("oscillation_base_speed", 9.0))
            amplitude = float(params.get("oscillation_amplitude", 4.0))
            period = max(float(params.get("oscillation_period", 3.0)), 0.2)
            target_speed = base_speed + amplitude * math.sin(2.0 * math.pi * step * float(dt) / period)
            return [{"target_speed": float(np.clip(target_speed, 0.0, 30.0))}]

        if behavior == "hard_brake":
            duration_steps = max(int(round(float(params.get("hard_brake_duration", 1.5)) / max(float(dt), 1e-6))), 1)
            trigger_step = int(params.setdefault("_trigger_step", int(step)))
            if int(step) - trigger_step <= duration_steps:
                return [{"target_speed": 0.0, "throttle": 0.0, "brake": float(params.get("hard_brake_strength", 1.0))}]
            return [{"target_speed": float(params.get("recovery_speed", 6.0))}]

        return [{"target_speed": float(params.get("decel_target_speed", 4.0))}]
