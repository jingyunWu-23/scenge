"""LC action decoding for straight-road templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from tests.scenarios.templates.straight_follow import FOLLOW_BEHAVIORS


def scale_action(value: float, low: float, high: float) -> float:
    clipped = float(np.clip(value, -1.0, 1.0))
    return float((clipped + 1.0) * 0.5 * (float(high) - float(low)) + float(low))


@dataclass
class StraightFollowParameterSpace:
    """Decode LC 4-D actions into straight-road template parameters."""

    ego_lane: int = 1
    ego_speed: float = 18.0

    def decode(self, action: Sequence[float], template_name: str = "straight_follow") -> Dict[str, float]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 4:
            raise ValueError(f"LC action must have at least 4 values, got {action.size}.")
        if template_name == "passing":
            return self._decode_passing(action)
        if template_name == "lane_change":
            return self._decode_lane_change(action)
        if template_name == "cut_in":
            return self._decode_cut_in(action)
        if template_name != "straight_follow":
            raise ValueError(f"Unsupported LC template: {template_name}")

        front_offset = scale_action(action[0], 12.0, 45.0)
        trigger_distance = scale_action(action[1], 8.0, 30.0)
        front_initial_speed = scale_action(action[2], 8.0, 22.0)
        behavior_strength = scale_action(action[3], 0.0, 1.0)

        behavior_index = min(int(behavior_strength * len(FOLLOW_BEHAVIORS)), len(FOLLOW_BEHAVIORS) - 1)
        behavior_type = FOLLOW_BEHAVIORS[behavior_index]

        params: Dict[str, float] = {
            "template_name": template_name,
            "behavior_type": behavior_type,
            "ego_lane": int(self.ego_lane),
            "ego_speed": float(self.ego_speed),
            "front_offset": front_offset,
            "front_initial_speed": front_initial_speed,
            "trigger_distance": trigger_distance,
            "behavior_strength": behavior_strength,
        }

        if behavior_type == "sudden_decel":
            params.update({
                "decel_target_speed": scale_action(action[3], 1.0, 8.0),
            })
        elif behavior_type == "speed_oscillation":
            params.update({
                "oscillation_base_speed": scale_action(action[2], 5.0, 14.0),
                "oscillation_amplitude": scale_action(action[3], 2.0, 7.0),
                "oscillation_period": scale_action(action[1], 1.0, 5.0),
            })
        elif behavior_type == "hard_brake":
            params.update({
                "hard_brake_strength": scale_action(action[3], 0.6, 1.0),
                "hard_brake_duration": scale_action(action[1], 0.5, 2.5),
                "recovery_speed": scale_action(action[2], 2.0, 8.0),
            })
        return params

    def _decode_passing(self, action: np.ndarray) -> Dict[str, float]:
        slow_front_offset = scale_action(action[0], 20.0, 45.0)
        target_lane_gap = scale_action(action[1], 15.0, 45.0)
        slow_front_speed = scale_action(action[2], 3.0, 10.0)
        target_lane_front_speed = scale_action(action[3], 6.0, 18.0)
        target_lane_front_offset = slow_front_offset + target_lane_gap
        return {
            "template_name": "passing",
            "behavior_type": "passing",
            "ego_lane": int(self.ego_lane),
            "target_lane": int(max(int(self.ego_lane) - 1, 0)),
            "ego_speed": float(self.ego_speed),
            "slow_front_offset": float(slow_front_offset),
            "slow_front_speed": float(slow_front_speed),
            "target_lane_front_offset": float(target_lane_front_offset),
            "target_lane_front_speed": float(target_lane_front_speed),
            "passing_gap": float(target_lane_gap),
        }

    def _decode_lane_change(self, action: np.ndarray) -> Dict[str, float]:
        blocker_offset = scale_action(action[0], 18.0, 42.0)
        target_gap = scale_action(action[1], 10.0, 34.0)
        blocker_speed = scale_action(action[2], 4.0, 12.0)
        rear_speed = scale_action(action[3], 16.0, 28.0)
        target_lane = int(max(int(self.ego_lane) - 1, 0))
        return {
            "template_name": "lane_change",
            "behavior_type": "right_escape",
            "ego_lane": int(self.ego_lane),
            "target_lane": target_lane,
            "ego_speed": float(self.ego_speed),
            "blocker_offset": float(blocker_offset),
            "blocker_speed": float(blocker_speed),
            "target_front_offset": float(blocker_offset + target_gap),
            "target_front_speed": float(scale_action(action[2], 8.0, 18.0)),
            "target_rear_offset": float(-scale_action(action[1], 6.0, 18.0)),
            "target_rear_speed": float(rear_speed),
            "target_lane_gap": float(target_gap),
        }

    def _decode_cut_in(self, action: np.ndarray) -> Dict[str, float]:
        cut_in_offset = scale_action(action[0], 10.0, 28.0)
        trigger_distance = scale_action(action[1], 12.0, 30.0)
        cut_in_speed = scale_action(action[2], 10.0, 22.0)
        lead_gap = scale_action(action[3], 10.0, 26.0)
        return {
            "template_name": "cut_in",
            "behavior_type": "cut_in_recovery",
            "ego_lane": int(self.ego_lane),
            "source_lane": int(min(int(self.ego_lane) + 1, 2)),
            "ego_speed": float(self.ego_speed),
            "cut_in_offset": float(cut_in_offset),
            "cut_in_speed": float(cut_in_speed),
            "cut_in_target_speed": float(scale_action(action[3], 4.0, 14.0)),
            "trigger_distance": float(trigger_distance),
            "lead_offset": float(cut_in_offset + lead_gap),
            "lead_speed": float(scale_action(action[2], 6.0, 16.0)),
            "lead_gap": float(lead_gap),
        }


def decode_lc_action(action: Sequence[float], template_name: str = "straight_follow") -> Dict[str, float]:
    return StraightFollowParameterSpace().decode(action, template_name=template_name)
