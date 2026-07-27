"""Straight-road lane-change template with adjacent-lane pressure."""

from __future__ import annotations

from typing import Dict, List

from tests.envs.scene_state import SceneState
from .base import StraightTemplate


class LaneChangeTemplate(StraightTemplate):
    name = "lane_change"

    def reset(self, decoded_params: Dict[str, float], seed: int = 0) -> SceneState:
        ego_lane = int(decoded_params.get("ego_lane", 1))
        target_lane = int(decoded_params.get("target_lane", max(ego_lane - 1, 0)))
        ego_speed = float(decoded_params.get("ego_speed", 18.0))

        blocker_offset = float(decoded_params.get("blocker_offset", 28.0))
        blocker_speed = float(decoded_params.get("blocker_speed", 7.0))
        target_front_offset = float(decoded_params.get("target_front_offset", 34.0))
        target_front_speed = float(decoded_params.get("target_front_speed", 12.0))
        target_rear_offset = float(decoded_params.get("target_rear_offset", -10.0))
        target_rear_speed = float(decoded_params.get("target_rear_speed", 22.0))

        ego = self._vehicle("ego", 0, route_s=0.0, speed=ego_speed, lane_id=ego_lane)
        blocker = self._vehicle("npc", 0, route_s=max(blocker_offset, 6.0), speed=blocker_speed, lane_id=ego_lane)
        target_front = self._vehicle(
            "npc", 1, route_s=max(target_front_offset, 6.0), speed=target_front_speed, lane_id=target_lane
        )
        target_rear = self._vehicle(
            "npc", 2, route_s=target_rear_offset, speed=target_rear_speed, lane_id=target_lane
        )
        return SceneState.from_vehicle_states(
            ego,
            [blocker, target_front, target_rear],
            route_length=self.route_length,
            template_name=self.name,
            decoded_params=dict(decoded_params),
            seed=int(seed),
            lane_width=self.lane_width,
            max_lanes=self.max_lanes,
        )

    def npc_intents(self, scene: SceneState, step: int, dt: float) -> List[Dict[str, float]]:
        params = scene.template.decoded_params
        return [
            {"target_speed": float(params.get("blocker_speed", 7.0))},
            {"target_speed": float(params.get("target_front_speed", 12.0))},
            {"target_speed": float(params.get("target_rear_speed", 22.0))},
        ]
