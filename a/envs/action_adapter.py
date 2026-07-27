"""Discrete action mapping for vehicle controls."""

from enum import IntEnum


class DiscreteDrivingAction(IntEnum):
    LANE_LEFT = 0
    KEEP_LANE = 1
    LANE_RIGHT = 2
    ACCELERATE = 3
    DECELERATE = 4


class ActionAdapter:
    """Converts discrete policy actions into high-level vehicle intents."""

    action_dim = 5

    def __init__(self, config=None):
        config = config or {}
        control = config.get("control", config)
        self.keep_lane_throttle = float(control.get("keep_lane_throttle", 0.45))
        self.accelerate_throttle = float(control.get("accelerate_throttle", 1.0))
        self.decelerate_brake = float(control.get("decelerate_brake", 0.55))
        self.lane_change_throttle = float(control.get("lane_change_throttle", 0.45))
        self.lane_change_steer = float(control.get("lane_change_steer", 0.35))
        self.delta_v = float(control.get("delta_v", 2.0))

    def to_control_intent_batch(self, actions) -> list:
        if isinstance(actions, (int, float)):
            actions = [actions]
        return [self.to_control_intent(action) for action in actions]

    def to_control_intent(self, action: int) -> dict:
        action = DiscreteDrivingAction(int(action))
        intent = {
            "action": int(action),
            "speed_delta": 0.0,
            "lane_delta": 0,
            "throttle": self.keep_lane_throttle,
            "brake": 0.0,
            "steer": 0.0,
            "lane_change": 0,
        }
        if action == DiscreteDrivingAction.ACCELERATE:
            intent.update({"speed_delta": self.delta_v, "throttle": self.accelerate_throttle})
        elif action == DiscreteDrivingAction.DECELERATE:
            intent.update({"speed_delta": -self.delta_v, "throttle": 0.0, "brake": self.decelerate_brake})
        elif action == DiscreteDrivingAction.LANE_LEFT:
            intent.update({"lane_delta": -1, "throttle": self.lane_change_throttle, "steer": -self.lane_change_steer, "lane_change": -1})
        elif action == DiscreteDrivingAction.LANE_RIGHT:
            intent.update({"lane_delta": 1, "throttle": self.lane_change_throttle, "steer": self.lane_change_steer, "lane_change": 1})
        return intent
