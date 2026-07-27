"""Straight-road LC passing template with a slow front NPC and target-lane NPC."""

from __future__ import annotations

from typing import Dict, List

from tests.envs.scene_state import SceneState
from .base import StraightTemplate


class PassingTemplate(StraightTemplate):
    name = "passing"

    def reset(self, decoded_params: Dict[str, float], seed: int = 0) -> SceneState:
        ego_lane = int(decoded_params.get("ego_lane", 1))
        target_lane = int(decoded_params.get("target_lane", max(ego_lane - 1, 0)))
        ego_speed = float(decoded_params.get("ego_speed", 18.0))

        slow_front_offset = float(decoded_params.get("slow_front_offset", 30.0))
        slow_front_speed = float(decoded_params.get("slow_front_speed", 6.0))
        target_lane_offset = float(decoded_params.get("target_lane_front_offset", 50.0))
        target_lane_speed = float(decoded_params.get("target_lane_front_speed", 12.0))

        ego = self._vehicle("ego", 0, route_s=0.0, speed=ego_speed, lane_id=ego_lane)
        slow_front = self._vehicle(
            "npc",
            0,
            route_s=max(slow_front_offset, 6.0),
            speed=slow_front_speed,
            lane_id=ego_lane,
        )
        target_lane_front = self._vehicle(
            "npc",
            1,
            route_s=max(target_lane_offset, 6.0),
            speed=target_lane_speed,
            lane_id=target_lane,
        )
        return SceneState.from_vehicle_states(
            ego,
            [slow_front, target_lane_front],
            route_length=self.route_length,
            template_name=self.name,
            decoded_params=dict(decoded_params),
            seed=int(seed),
            lane_width=self.lane_width,
            max_lanes=self.max_lanes,
        )

    def npc_intents(self, scene: SceneState, step: int, dt: float) -> List[Dict[str, float]]:
        params = scene.template.decoded_params
        slow_front_speed = float(params.get("slow_front_speed", 6.0))
        target_lane_speed = float(params.get("target_lane_front_speed", 12.0))
        return [
            {"target_speed": slow_front_speed},
            {"target_speed": target_lane_speed},
        ]
