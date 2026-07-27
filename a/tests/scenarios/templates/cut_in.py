"""Straight-road cut-in template where an adjacent NPC inserts ahead of ego."""

from __future__ import annotations

from typing import Dict, List

from tests.envs.scene_state import SceneState
from .base import StraightTemplate


class CutInTemplate(StraightTemplate):
    name = "cut_in"

    def reset(self, decoded_params: Dict[str, float], seed: int = 0) -> SceneState:
        ego_lane = int(decoded_params.get("ego_lane", 1))
        source_lane = int(decoded_params.get("source_lane", min(ego_lane + 1, self.max_lanes - 1)))
        ego_speed = float(decoded_params.get("ego_speed", 18.0))

        cut_in_offset = float(decoded_params.get("cut_in_offset", 16.0))
        cut_in_speed = float(decoded_params.get("cut_in_speed", 16.0))
        lead_offset = float(decoded_params.get("lead_offset", 42.0))
        lead_speed = float(decoded_params.get("lead_speed", 10.0))

        ego = self._vehicle("ego", 0, route_s=0.0, speed=ego_speed, lane_id=ego_lane)
        cutter = self._vehicle("npc", 0, route_s=max(cut_in_offset, 6.0), speed=cut_in_speed, lane_id=source_lane)
        lead = self._vehicle("npc", 1, route_s=max(lead_offset, cut_in_offset + 8.0), speed=lead_speed, lane_id=ego_lane)
        return SceneState.from_vehicle_states(
            ego,
            [cutter, lead],
            route_length=self.route_length,
            template_name=self.name,
            decoded_params=dict(decoded_params),
            seed=int(seed),
            lane_width=self.lane_width,
            max_lanes=self.max_lanes,
        )

    def npc_intents(self, scene: SceneState, step: int, dt: float) -> List[Dict[str, float]]:
        params = scene.template.decoded_params
        ego = scene.ego_vehicle
        cutter = scene.adv_cav_vehicles[0] if scene.adv_cav_vehicles else None
        if ego is None or cutter is None:
            return []

        gap = float((cutter.route_s or cutter.x) - (ego.route_s or ego.x))
        trigger_distance = float(params.get("trigger_distance", 22.0))
        triggered = gap <= trigger_distance or bool(params.get("_triggered", False))
        params["_triggered"] = triggered

        lane_delta = 0
        if triggered and int(cutter.lane_id) != int(params.get("ego_lane", 1)):
            lane_delta = -1 if int(cutter.lane_id) > int(params.get("ego_lane", 1)) else 1

        return [
            {
                "target_speed": float(params.get("cut_in_target_speed", params.get("cut_in_speed", 16.0))),
                "lane_delta": lane_delta,
            },
            {"target_speed": float(params.get("lead_speed", 10.0))},
        ]
