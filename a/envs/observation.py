"""Low-dimensional observation builder for the first migration phase."""

import numpy as np

from .vehicle_state import longitudinal_gap


class LowDimObservationBuilder:
    """Builds 25-D observations with explicit adjacent-lane structure."""

    def __init__(self, state_dim: int = 25, max_neighbors: int = 1, lane_width: float = 4.0, max_lanes: int = 4):
        self.state_dim = state_dim
        self.max_neighbors = max_neighbors
        self.lane_width = float(lane_width)
        self.max_lanes = int(max_lanes)

    def build_multi_agent_observation(self, scenario) -> np.ndarray:
        agents = list(getattr(scenario, "adv_cav_vehicles", []))
        if not agents and scenario.ego_vehicle is not None:
            agents = [scenario.ego_vehicle]
        return np.asarray([self.build_vehicle_observation(scenario, vehicle) for vehicle in agents], dtype=np.float32)

    def build_ego_observation(self, scenario) -> np.ndarray:
        if scenario.ego_vehicle is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        return self.build_vehicle_observation(scenario, scenario.ego_vehicle)

    def build_hdv_observations(self, scenario):
        return [self.build_matrix_observation(scenario, vehicle) for vehicle in getattr(scenario, "hdv_vehicles", [])]

    def build_matrix_observation(self, scenario, vehicle) -> np.ndarray:
        flat = self.build_vehicle_observation(scenario, vehicle)
        return flat.reshape((5, 5))

    def build_vehicle_observation(self, scenario, vehicle) -> np.ndarray:
        if (
            not getattr(vehicle, "active", True)
            or not getattr(vehicle, "physical_active", True)
        ):
            return np.zeros(self.state_dim, dtype=np.float32)
        lane_norm = float(vehicle.lane_id) / max(self.max_lanes - 1, 1)
        route_s = vehicle.route_s if vehicle.route_s is not None else vehicle.x
        values = [
            np.clip(route_s / max(scenario.route_length, 1e-6), -0.5, 1.5),
            np.clip(vehicle.lateral_offset / max(self.lane_width, 1e-6), -1.5, 1.5),
            np.clip(vehicle.speed / 35.0, 0.0, 1.5),
            np.clip(vehicle.heading_error / 3.1415926, -1.0, 1.0),
            np.clip(lane_norm, 0.0, 1.0),
            np.clip(float(scenario.route_completion), 0.0, 1.0),
        ]

        same = self._lane_context(scenario, vehicle, vehicle.lane_id)
        left_available = self._lane_available(vehicle.lane_id - 1)
        left = self._lane_context(scenario, vehicle, vehicle.lane_id - 1)
        right_available = self._lane_available(vehicle.lane_id + 1)
        right = self._lane_context(scenario, vehicle, vehicle.lane_id + 1)

        values.extend([
            same["front_gap"], same["front_dv"], same["front_ttc"],
            left_available, left["front_gap"], left["rear_gap"], left["front_dv"], left["rear_dv"],
            right_available, right["front_gap"], right["rear_gap"], right["front_dv"], right["rear_dv"],
        ])

        risk = self.compute_lane_risk_summary(scenario, vehicle, same, left, right, left_available, right_available)
        values.extend([
            risk["front_risk"],
            risk["rear_risk"],
            risk["left_risk"],
            risk["right_risk"],
            risk["escape_lane_available"],
            risk["is_surrounded"],
        ])

        while len(values) < self.state_dim:
            values.append(0.0)
        return np.asarray(values[:self.state_dim], dtype=np.float32)


    def compute_lane_risk_summary(self, scenario, vehicle, same, left, right, left_available, right_available):
        front_risk = self._risk_from_gap_ttc(same["front_gap"], same["front_ttc"])
        rear_risk = self._risk_from_gap_ttc(same["rear_gap"], 1.0)
        left_risk = 0.5 * self._risk_from_gap_ttc(left["front_gap"], left["front_ttc"]) + 0.5 * self._risk_from_gap_ttc(left["rear_gap"], 1.0)
        right_risk = 0.5 * self._risk_from_gap_ttc(right["front_gap"], right["front_ttc"]) + 0.5 * self._risk_from_gap_ttc(right["rear_gap"], 1.0)
        lane_safe_threshold = 0.35
        escape_lane_available = float(
            (left_available > 0.5 and left_risk < lane_safe_threshold)
            or (right_available > 0.5 and right_risk < lane_safe_threshold)
            or front_risk < lane_safe_threshold
        )
        risk_threshold = 0.55
        is_surrounded = float(
            front_risk > risk_threshold
            and rear_risk > risk_threshold
            and (left_available < 0.5 or left_risk > risk_threshold)
            and (right_available < 0.5 or right_risk > risk_threshold)
        )
        return {
            "front_risk": float(front_risk),
            "rear_risk": float(rear_risk),
            "left_risk": float(left_risk),
            "right_risk": float(right_risk),
            "mean_risk": float(np.mean([front_risk, rear_risk, left_risk, right_risk])),
            "max_risk": float(max(front_risk, rear_risk, left_risk, right_risk)),
            "escape_lane_available": escape_lane_available,
            "is_surrounded": is_surrounded,
        }

    @staticmethod
    def _risk_from_gap_ttc(norm_gap: float, norm_ttc: float):
        distance_risk = max(0.0, 1.0 - float(norm_gap) / 0.25)
        ttc_risk = max(0.0, 1.0 - float(norm_ttc) / 0.5)
        return float(np.clip(0.6 * distance_risk + 0.4 * ttc_risk, 0.0, 1.0))

    def _lane_context(self, scenario, vehicle, lane_id: int):
        front = []
        rear = []
        for other in scenario.vehicles:
            if (
                other is vehicle
                or other.lane_id != lane_id
                or int(other.road_id) != int(vehicle.road_id)
                or int(other.section_id) != int(vehicle.section_id)
                or not getattr(other, "physical_active", True)
            ):
                continue
            gap = longitudinal_gap(vehicle, other)
            if gap >= 0:
                front.append((gap, other))
            else:
                rear.append((-gap, other))
        front_gap, front_vehicle = min(front, key=lambda item: item[0]) if front else (100.0, None)
        rear_gap, rear_vehicle = min(rear, key=lambda item: item[0]) if rear else (100.0, None)
        return {
            "front_gap": self._norm_gap(front_gap, 100.0),
            "rear_gap": self._norm_gap(rear_gap, 100.0),
            "front_dv": self._norm_dv(front_vehicle.speed - vehicle.speed) if front_vehicle else 0.0,
            "rear_dv": self._norm_dv(rear_vehicle.speed - vehicle.speed) if rear_vehicle else 0.0,
            "front_ttc": self._ttc(vehicle, front_vehicle) if front_vehicle else 1.0,
        }

    def _lane_available(self, lane_id: int):
        return 1.0 if 0 <= int(lane_id) < self.max_lanes else 0.0

    def _relative_features(self, vehicle, other):
        dx = np.clip(longitudinal_gap(vehicle, other) / 100.0, -1.0, 1.0)
        dy = np.clip((other.y - vehicle.y) / (self.lane_width * 3.0), -1.0, 1.0)
        dv = self._norm_dv(other.speed - vehicle.speed)
        same_lane = 1.0 if other.lane_id == vehicle.lane_id else 0.0
        ttc = self._ttc(vehicle, other)
        return [dx, dy, dv, same_lane, ttc]

    @staticmethod
    def _nearest_neighbors(scenario, vehicle):
        others = [
            other for other in scenario.vehicles
            if other is not vehicle and getattr(other, "physical_active", True)
        ]
        return sorted(others, key=lambda other: abs(longitudinal_gap(vehicle, other)))

    @staticmethod
    def _norm_gap(value: float, scale: float):
        return float(np.clip(value / max(scale, 1e-6), -1.0, 1.0))

    @staticmethod
    def _norm_dv(value: float):
        return float(np.clip(value / 35.0, -1.0, 1.0))

    @staticmethod
    def _ttc(vehicle, other) -> float:
        if other is None:
            return 1.0
        gap = longitudinal_gap(vehicle, other)
        if gap <= 0.0:
            return 1.0
        closing_speed = vehicle.speed - other.speed
        if closing_speed <= 1e-6:
            return 1.0
        ttc = gap / closing_speed
        return float(np.clip(ttc / 10.0, 0.0, 1.0))
