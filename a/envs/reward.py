"""Reward manager for ego, adversarial CAV, and HDV roles."""

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .vehicle_state import longitudinal_gap


@dataclass
class RewardResult:
    reward: float
    info: Dict = field(default_factory=dict)


class RiskFieldReward:
    """Risk-field adversarial reward migrated from the highway-env baseline."""

    def __init__(self):
        self.previous_potential = {}
        self.previous_ego_speed = None
        self.previous_ego_lane_id = None
        self.speed_match_weight = 1.0
        self.behavior_change_weight = 1.0
        self.interaction_variance_weight = 0.5
        self.diversity_weight = 1.0
        self.adv_close_distance = 10.0
        self.adv_close_penalty_weight = 1.0
        self.adv_ttc_threshold = 3.0
        self.adv_ttc_penalty_weight = 2.0
        self.duplicate_mode_distance = 15.0
        self.duplicate_mode_penalty_weight = 0.5
        self.collision_entry_penalty = 40.0
        self.collision_contact_penalty = 2.0
        self.collision_recovery_reward = 5.0

    def reset(self):
        self.previous_potential = {}
        self.previous_ego_speed = None
        self.previous_ego_lane_id = None

    def compute_agent_rewards(self, scenario) -> tuple:
        rewards = []
        potentials = []
        deltas = []
        speed_matches = []
        close_penalties = []
        ttc_penalties = []
        duplicate_penalties = []
        min_adv_distances = []
        min_adv_ttcs = []
        ego = scenario.ego_vehicle
        behavior_change, behavior_info = self._ego_behavior_change(ego)
        interaction_stats = self._interaction_diversity(scenario)
        interaction_modes = interaction_stats["modes"]
        interaction_coverage = interaction_stats["coverage"]
        interaction_entropy = interaction_stats["entropy"]
        diversity_reward = self.diversity_weight * interaction_entropy
        active_advs = [
            vehicle for vehicle in scenario.adv_cav_vehicles
            if getattr(vehicle, "active", True) and getattr(vehicle, "physical_active", True)
        ]
        newly_collided = set(getattr(scenario, "newly_collided_adv_indices", ()))
        colliding = set(getattr(scenario, "colliding_adv_indices", ()))
        recovered = set(getattr(scenario, "recovered_adv_indices", ()))

        for vehicle in scenario.adv_cav_vehicles:
            is_active = (
                getattr(vehicle, "active", True)
                and getattr(vehicle, "physical_active", True)
            )
            potential = self._agent_potential(scenario, vehicle) if is_active else 0.0
            previous = self.previous_potential.get(vehicle.index, potential)
            delta = self._shape_delta(previous - potential) if is_active else 0.0
            if is_active:
                self.previous_potential[vehicle.index] = potential
            else:
                self.previous_potential.pop(vehicle.index, None)

            if ego is not None and ego.crashed:
                crash_reward = 100.0
            elif vehicle.index in newly_collided:
                crash_reward = -self.collision_entry_penalty
            elif vehicle.index in colliding:
                crash_reward = -self.collision_contact_penalty
            elif vehicle.index in recovered:
                crash_reward = self.collision_recovery_reward
            else:
                crash_reward = 0.0

            distance = (
                float(np.linalg.norm(vehicle.position - ego.position))
                if ego is not None and is_active else 0.0
            )
            distance_weight = float(max(0.1, np.exp(-distance / 40.0))) if is_active else 0.0
            shaped_delta = delta * distance_weight
            speed_match = self._speed_match_reward(vehicle, ego) if is_active else 0.0
            interaction_bonus = self.interaction_variance_weight * interaction_coverage if is_active else 0.0
            behavior_bonus = self.behavior_change_weight * behavior_change if is_active else 0.0
            safety = self._adv_safety_penalty(vehicle, active_advs, ego) if is_active else {
                "close": 0.0, "ttc": 0.0, "duplicate": 0.0,
                "min_distance": -1.0, "min_ttc": -1.0,
            }

            reward = (
                10.0 * shaped_delta
                + crash_reward
                + self.speed_match_weight * speed_match
                + interaction_bonus
                + behavior_bonus
                + (diversity_reward if is_active else 0.0)
                - safety["close"]
                - safety["ttc"]
                - safety["duplicate"]
            )
            rewards.append(float(reward))
            potentials.append(float(potential))
            deltas.append(float(previous - potential) if is_active else 0.0)
            speed_matches.append(float(speed_match))
            close_penalties.append(float(safety["close"]))
            ttc_penalties.append(float(safety["ttc"]))
            duplicate_penalties.append(float(safety["duplicate"]))
            min_adv_distances.append(float(safety["min_distance"]))
            min_adv_ttcs.append(float(safety["min_ttc"]))

        self._update_previous_ego(ego)
        info = {
            "adv_speed_match_rewards": tuple(speed_matches),
            "adv_close_distance_penalties": tuple(close_penalties),
            "adv_ttc_penalties": tuple(ttc_penalties),
            "adv_duplicate_mode_penalties": tuple(duplicate_penalties),
            "adv_min_team_distances": tuple(min_adv_distances),
            "adv_min_team_ttcs": tuple(min_adv_ttcs),
            "adv_interaction_variance": float(interaction_coverage),
            "adv_interaction_coverage": float(interaction_coverage),
            "adv_interaction_entropy": float(interaction_entropy),
            "adv_interaction_modes": tuple(interaction_modes),
            "adv_diversity_reward": float(diversity_reward),
            "adv_active_count": int(len(active_advs)),
            "adv_ego_behavior_change": float(behavior_change),
            **behavior_info,
        }
        return tuple(rewards), tuple(potentials), tuple(deltas), info

    def _adv_safety_penalty(self, vehicle, active_advs, ego):
        others = [other for other in active_advs if other is not vehicle]
        if not others:
            return {"close": 0.0, "ttc": 0.0, "duplicate": 0.0, "min_distance": -1.0, "min_ttc": -1.0}

        distances = [float(np.linalg.norm(vehicle.position - other.position)) for other in others]
        min_distance = min(distances)
        close_penalty = self.adv_close_penalty_weight * max(
            0.0, 1.0 - min_distance / max(self.adv_close_distance, 1e-6)
        )

        ttc_values = []
        for other in others:
            if other.lane_id != vehicle.lane_id:
                continue
            gap = longitudinal_gap(vehicle, other)
            if gap >= 0.0:
                closing_speed = float(vehicle.speed - other.speed)
            else:
                closing_speed = float(other.speed - vehicle.speed)
            if closing_speed > 1e-6:
                ttc_values.append(abs(float(gap)) / closing_speed)
        min_ttc = min(ttc_values) if ttc_values else -1.0
        ttc_penalty = 0.0
        if 0.0 <= min_ttc < self.adv_ttc_threshold:
            ttc_penalty = self.adv_ttc_penalty_weight * (
                1.0 - min_ttc / max(self.adv_ttc_threshold, 1e-6)
            )

        duplicate_count = 0
        if ego is not None:
            mode = self._interaction_mode(ego, vehicle)
            for other, distance in zip(others, distances):
                if distance < self.duplicate_mode_distance and self._interaction_mode(ego, other) == mode:
                    duplicate_count += 1
        duplicate_penalty = self.duplicate_mode_penalty_weight * min(duplicate_count, 1)
        return {
            "close": float(close_penalty),
            "ttc": float(ttc_penalty),
            "duplicate": float(duplicate_penalty),
            "min_distance": float(min_distance),
            "min_ttc": float(min_ttc),
        }

    def _ego_behavior_change(self, ego):
        if ego is None or self.previous_ego_speed is None:
            return 0.0, {
                "adv_ego_speed_change": 0.0,
                "adv_ego_deceleration": 0.0,
                "adv_ego_lane_change": 0.0,
            }
        speed_change = abs(float(ego.speed) - float(self.previous_ego_speed))
        deceleration = max(0.0, float(self.previous_ego_speed) - float(ego.speed))
        lane_change = float(int(int(ego.lane_id) != int(self.previous_ego_lane_id)))
        normalized_speed_change = min(speed_change / 8.0, 1.0)
        normalized_deceleration = min(deceleration / 6.0, 1.0)
        behavior = float(np.clip(0.5 * normalized_speed_change + 0.5 * normalized_deceleration + lane_change, 0.0, 2.0))
        return behavior, {
            "adv_ego_speed_change": float(speed_change),
            "adv_ego_deceleration": float(deceleration),
            "adv_ego_lane_change": float(lane_change),
        }

    def _update_previous_ego(self, ego):
        if ego is None:
            self.previous_ego_speed = None
            self.previous_ego_lane_id = None
            return
        self.previous_ego_speed = float(ego.speed)
        self.previous_ego_lane_id = int(ego.lane_id)

    def _speed_match_reward(self, vehicle, ego):
        if ego is None:
            return 0.0
        return float(np.exp(-abs(float(vehicle.speed) - float(ego.speed)) / 5.0))

    def _interaction_diversity(self, scenario):
        ego = scenario.ego_vehicle
        advs = [
            adv for adv in scenario.adv_cav_vehicles
            if getattr(adv, "active", True) and getattr(adv, "physical_active", True)
        ]
        if ego is None or not advs:
            return {"coverage": 0.0, "entropy": 0.0, "modes": []}
        modes = [self._interaction_mode(ego, adv) for adv in advs]
        coverage = float(len(set(modes)) / max(len(modes), 1))
        entropy = self._normalized_entropy(modes)
        return {"coverage": coverage, "entropy": entropy, "modes": modes}

    @staticmethod
    def _normalized_entropy(modes):
        if not modes:
            return 0.0
        _, counts = np.unique(np.asarray(modes, dtype=object), return_counts=True)
        probabilities = counts.astype(float) / max(float(np.sum(counts)), 1.0)
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-8)))
        return float(entropy / max(math.log(max(len(counts), 2)), 1e-6))

    @staticmethod
    def _interaction_mode(ego, adv):
        longitudinal = longitudinal_gap(ego, adv)
        lane_delta = int(adv.lane_id) - int(ego.lane_id)
        if abs(longitudinal) <= 8.0 and lane_delta != 0:
            return "side_by_side"
        if lane_delta == 0 and longitudinal > 0.0:
            return "front_same_lane"
        if lane_delta == 0:
            return "rear_same_lane"
        if longitudinal > 0.0:
            return "front_adjacent"
        return "rear_adjacent"

    def _agent_potential(self, scenario, vehicle):
        ego = scenario.ego_vehicle
        if ego is None:
            return 0.0
        other_advs = [
            other for other in scenario.adv_cav_vehicles
            if (
                other is not vehicle
                and getattr(other, "active", True)
                and getattr(other, "physical_active", True)
            )
        ]
        active_hdvs = [
            other for other in scenario.hdv_vehicles
            if getattr(other, "active", True) and not other.crashed
        ]
        hdv_vehicle = active_hdvs[0] if active_hdvs else ego
        p1_vehicle = other_advs[0] if len(other_advs) > 0 else ego
        p2_vehicle = other_advs[1] if len(other_advs) > 1 else hdv_vehicle
        return self._risk_field(
            self._xy(vehicle), self._xy(ego), self._xy(p1_vehicle), self._xy(p2_vehicle), self._xy(hdv_vehicle),
            vehicle.speed, p1_vehicle.speed, p2_vehicle.speed, hdv_vehicle.speed, ego.speed,
        )

    @staticmethod
    def _shape_delta(delta: float) -> float:
        delta = max(-1.0, min(1.0, float(delta)))
        if delta < 0.0:
            delta = 0.5 * delta
        return delta

    @staticmethod
    def _xy(vehicle):
        route_s = vehicle.route_s if vehicle.route_s is not None else vehicle.x
        return np.asarray([route_s, vehicle.lateral_offset], dtype=float)

    @staticmethod
    def _risk_field(p0, pg, p1, p2, pz, v0, v1, v2, vz, vg):
        m = 1560.0
        length = 5.0
        width = 2.0
        max_brake_force = 3000.0
        pt = 0.01

        y, x = p0[0], p0[1]
        y0, x0 = p1[0], p1[1]
        y1, x1 = p2[0], p2[1]
        yg, xg = pg[0], pg[1]
        yz, xz = pz[0], pz[1]

        db = RiskFieldReward._braking_distance(m, length, max_brake_force, v0, v1)
        dt = db + 10.0
        c2 = -np.log(pt) / max(db ** 2, 1e-6)
        c1 = RiskFieldReward._lateral_coefficient(y, y0, db, c2, width, pt)

        db1 = RiskFieldReward._braking_distance(m, length, max_brake_force, v0, v2)
        dt1 = db1 + 10.0
        c21 = -np.log(pt) / max(db1 ** 2, 1e-6)
        c11 = RiskFieldReward._lateral_coefficient(y, y1, db1, c21, width, pt)

        dbz = RiskFieldReward._braking_distance(m, length, max_brake_force, v0, vz)
        dtz = dbz + 10.0
        cz2 = -np.log(pt) / max(dbz ** 2, 1e-6)
        cz1 = RiskFieldReward._lateral_coefficient(y, yz, dbz, cz2, width, pt)

        az = np.exp(-(x - 2.0) ** 2)
        az1 = np.exp(-(x - 6.0) ** 2)
        ay = RiskFieldReward._distance_modifier(y, y0, db, dt)
        ay1 = RiskFieldReward._distance_modifier(y, y1, db1, dt1)
        ayz = RiskFieldReward._distance_modifier(y, yz, dbz, dtz)
        pr = az * ay * ay1 * ayz + az1 * ay * ay1 * ayz

        future_ego_y = yg + vg * 0.2
        distance_to_future_ego = math.sqrt((future_ego_y - y) ** 2 + (xg - x) ** 2)
        if distance_to_future_ego < 20.0:
            ps = distance_to_future_ego ** 2 / 150.0
        else:
            ps = abs(yg - y) ** 2 / 150.0
        if distance_to_future_ego < 6.0:
            ps = 6.0 ** 2 / 150.0

        po = abs(np.exp(-c1 * (x - x0) ** 2 - c2 * (y - y0) ** 2) - pt) / (1.0 - pt)
        p_1 = abs(np.exp(-c11 * (x - x1) ** 2 - c21 * (y - y1) ** 2) - pt) / (1.0 - pt)
        pz_value = abs(np.exp(-cz1 * (x - xz) ** 2 - cz2 * (y - yz) ** 2) - pt) / (1.0 - pt)
        return float(p_1 + po + ps + 0.1 * pr + pz_value)

    @staticmethod
    def _braking_distance(m, length, max_brake_force, ego_speed, obstacle_speed):
        if obstacle_speed >= ego_speed:
            return 5.0
        return m * (ego_speed ** 2 - obstacle_speed ** 2) / (8.0 * max_brake_force) + length / 2.0

    @staticmethod
    def _lateral_coefficient(y, obstacle_y, braking_distance, c2, width, pt):
        if abs(y - obstacle_y) > braking_distance:
            return 0.0
        denom = (width * (np.sin((y - obstacle_y - braking_distance) / braking_distance * np.pi - np.pi / 2.0) + 2.0)) ** 2 * 0.4
        return float(-4.0 * (np.log(pt) + c2 * (y - obstacle_y) ** 2) / max(denom, 1e-6))

    @staticmethod
    def _distance_modifier(y, obstacle_y, braking_distance, danger_distance):
        distance = abs(y - obstacle_y)
        if distance <= braking_distance:
            return 0.0
        if distance >= danger_distance:
            return 1.0
        return (distance - braking_distance) / max(danger_distance - braking_distance, 1e-6)


class CatSafetyEgoReward:
    """Ego reward aligned with MARL1/ego/reward/cat_reward.py::CatSafetyReward."""

    def __init__(
        self,
        driving_reward: float = 1.0,
        speed_reward: float = 0.1,
        success_reward: float = 30.0,
        out_of_road_penalty: float = 5.0,
        crash_vehicle_penalty: float = 20.0,
        crash_object_penalty: float = 5.0,
        use_lateral_reward: bool = False,
        lane_width: float = 4.0,
        max_speed: float = 30.0,
        route_completion_threshold: float = 0.95,
        alpha: float = 1.0,
        safe_distance: float = 25.0,
        base_risk_tau: float = 0.25,
        tau_entropy_weight: float = 0.20,
        tau_coverage_weight: float = 0.15,
        tau_performance_weight: float = 0.20,
        risk_violation_weight: float = 1.0,
        lane_change_bonus: float = 0.15,
        lane_change_cost: float = 0.10,
        oscillation_penalty: float = 0.35,
        oscillation_window: int = 30,
        front_gap_penalty: float = 0.5,
        blocked_front_distance: float = 18.0,
        stop_speed_threshold: float = 2.0,
        stop_penalty: float = 0.6,
        obstacle_penalty_weight: float = 2.0,
        obstacle_escape_bonus: float = 0.5,
        obstacle_escape_threshold: float = 0.35,
        obstacle_escape_margin: float = 0.15,
        obstacle_recovery_bonus: float = 0.3,
        obstacle_recovery_speed: float = 6.0,
        obstacle_recovery_low_speed: float = 2.5,
        obstacle_recovery_risk_threshold: float = 0.25,
        obstacle_recovery_clear_risk: float = 0.10,
        obstacle_recovery_cooldown: int = 50,
        hazardous_lane_change_penalty: float = 1.0,
        hazardous_lane_change_risk_threshold: float = 0.25,
        obstacle_takeover_penalty: float = 0.02,
        obstacle_takeover_release_bonus: float = 0.2,
    ):
        self.driving_reward = float(driving_reward)
        self.speed_reward = float(speed_reward)
        self.success_reward = float(success_reward)
        self.out_of_road_penalty = float(out_of_road_penalty)
        self.crash_vehicle_penalty = float(crash_vehicle_penalty)
        self.crash_object_penalty = float(crash_object_penalty)
        self.use_lateral_reward = bool(use_lateral_reward)
        self.lane_width = float(lane_width)
        self.max_speed = float(max_speed)
        self.route_completion_threshold = float(route_completion_threshold)
        self.alpha = float(alpha)
        self.safe_distance = float(safe_distance)
        self.base_risk_tau = float(base_risk_tau)
        self.tau_entropy_weight = float(tau_entropy_weight)
        self.tau_coverage_weight = float(tau_coverage_weight)
        self.tau_performance_weight = float(tau_performance_weight)
        self.risk_violation_weight = float(risk_violation_weight)
        self.lane_change_bonus = float(lane_change_bonus)
        self.lane_change_cost = float(lane_change_cost)
        self.oscillation_penalty = float(oscillation_penalty)
        self.oscillation_window = int(oscillation_window)
        self.front_gap_penalty = float(front_gap_penalty)
        self.blocked_front_distance = float(blocked_front_distance)
        self.stop_speed_threshold = float(stop_speed_threshold)
        self.stop_penalty = float(stop_penalty)
        self.obstacle_penalty_weight = float(obstacle_penalty_weight)
        self.obstacle_escape_bonus = float(obstacle_escape_bonus)
        self.obstacle_escape_threshold = float(obstacle_escape_threshold)
        self.obstacle_escape_margin = float(obstacle_escape_margin)
        self.obstacle_recovery_bonus = float(obstacle_recovery_bonus)
        self.obstacle_recovery_speed = float(obstacle_recovery_speed)
        self.obstacle_recovery_low_speed = float(obstacle_recovery_low_speed)
        self.obstacle_recovery_risk_threshold = float(obstacle_recovery_risk_threshold)
        self.obstacle_recovery_clear_risk = float(obstacle_recovery_clear_risk)
        self.obstacle_recovery_cooldown = int(obstacle_recovery_cooldown)
        self.hazardous_lane_change_penalty = float(hazardous_lane_change_penalty)
        self.hazardous_lane_change_risk_threshold = float(hazardous_lane_change_risk_threshold)
        self.obstacle_takeover_penalty = float(obstacle_takeover_penalty)
        self.obstacle_takeover_release_bonus = float(obstacle_takeover_release_bonus)
        self._last_long = None
        self._last_lane_index = None
        self._last_obstacle_risk = 0.0
        self._obstacle_low_speed_pending = False
        self._obstacle_recovery_cooldown_steps = 1000000
        self._steps_since_lane_change = 1000000

    def reset(self):
        self._last_long = None
        self._last_lane_index = None
        self._last_obstacle_risk = 0.0
        self._obstacle_low_speed_pending = False
        self._obstacle_recovery_cooldown_steps = 1000000
        self._steps_since_lane_change = 1000000

    def compute(self, scenario):
        ego = scenario.ego_vehicle
        if ego is None:
            return 0.0, {
                "ego_driving_reward": 0.0,
                "ego_speed_reward": 0.0,
                "ego_risk_penalty": 0.0,
                "ego_terminal_reward": 0.0,
            }

        done = bool(scenario.crash or scenario.success or not ego.on_road)
        if done:
            terminal, terminal_type = self._terminal_reward(scenario, ego)
            return terminal, {
                "ego_driving_reward": 0.0,
                "ego_speed_reward": 0.0,
                "ego_risk_penalty": 0.0,
                "ego_terminal_reward": terminal,
                "ego_terminal_type": terminal_type,
                "ego_route_completion": float(scenario.route_completion),
            }

        previous_lane_index = self._last_lane_index
        driving, driving_info = self._driving_reward(ego)
        speed, speed_info = self._speed_reward(ego)
        risk, risk_info = self._risk_penalty(scenario, ego)
        lane_bonus, lane_info = self._lane_safety_shaping(scenario, ego, previous_lane_index)
        stop_penalty, stop_info = self._stop_penalty(scenario, ego)
        obstacle_penalty, obstacle_bonus, obstacle_info = self._obstacle_shaping(ego, previous_lane_index)
        reward = driving + speed - risk + lane_bonus + obstacle_bonus - stop_penalty - obstacle_penalty
        return float(reward), {
            "ego_driving_reward": float(driving),
            "ego_speed_reward": float(speed),
            "ego_risk_penalty": float(-risk),
            "ego_lane_change_bonus": float(lane_bonus),
            "ego_stop_penalty": float(-stop_penalty),
            "ego_obstacle_penalty": float(-obstacle_penalty),
            "ego_obstacle_escape_bonus": float(obstacle_info.get("ego_obstacle_escape_bonus_value", 0.0)),
            "ego_obstacle_recovery_bonus": float(obstacle_info.get("ego_obstacle_recovery_bonus_value", 0.0)),
            "ego_terminal_reward": 0.0,
            **driving_info,
            **speed_info,
            **risk_info,
            **lane_info,
            **stop_info,
            **obstacle_info,
        }

    def _stop_penalty(self, scenario, ego):
        front_gap = self._front_gap(scenario, ego, ego.lane_id)
        front_blocked = front_gap <= self.blocked_front_distance
        info = {
            "ego_stop_penalty_value": 0.0,
            "ego_stop_front_gap": float(front_gap),
            "ego_stop_front_blocked": bool(front_blocked),
        }
        if self.stop_speed_threshold <= 0.0 or ego.speed >= self.stop_speed_threshold:
            return 0.0, info
        if front_blocked:
            return 0.0, info
        penalty = self.stop_penalty * (1.0 - ego.speed / max(self.stop_speed_threshold, 1e-6))
        info["ego_stop_penalty_value"] = float(penalty)
        return float(penalty), info

    def _driving_reward(self, ego):
        long_now = float(ego.route_s if ego.route_s is not None else ego.x)
        lateral_now = float(ego.lateral_offset)
        if self._last_long is None:
            self._last_long = long_now
            self._last_lane_index = int(ego.lane_id)
            return 0.0, {"ego_forward_progress": 0.0, "ego_lateral_offset": lateral_now, "ego_lateral_factor": 1.0}

        if int(ego.lane_id) != int(self._last_lane_index):
            self._last_long = long_now
            self._last_lane_index = int(ego.lane_id)
            return 0.0, {"ego_forward_progress": 0.0, "ego_lateral_offset": lateral_now, "ego_lateral_factor": 1.0}

        forward_progress = float(np.clip(long_now - self._last_long, -5.0, 10.0))
        self._last_long = long_now
        if self.use_lateral_reward:
            lateral_factor = float(np.clip(1.0 - 2.0 * abs(lateral_now) / self.lane_width, 0.0, 1.0))
        else:
            lateral_factor = 1.0
        driving = self.driving_reward * forward_progress * lateral_factor
        return float(driving), {
            "ego_forward_progress": forward_progress,
            "ego_route_s": long_now,
            "ego_lateral_offset": lateral_now,
            "ego_heading_error": float(ego.heading_error),
            "ego_lateral_factor": lateral_factor,
        }

    def _speed_reward(self, ego):
        speed_ratio = float(ego.speed / max(self.max_speed, 1e-6))
        speed = self.speed_reward * speed_ratio
        return float(speed), {"ego_speed_ratio": speed_ratio}

    def _terminal_reward(self, scenario, ego):
        route_completion = float(scenario.route_completion)
        if not ego.on_road:
            return -self.out_of_road_penalty, "out_of_road"
        if ego.crashed:
            return -self.crash_vehicle_penalty, "crash_vehicle"
        if route_completion >= self.route_completion_threshold:
            return self.success_reward, "success"
        return 0.0, "timeout"

    def _risk_penalty(self, scenario, ego):
        raw_risk = 0.0
        min_distance = float("inf")
        ego_pos = np.asarray(ego.position, dtype=float)
        for adv_vehicle in scenario.adv_cav_vehicles:
            if (
                not getattr(adv_vehicle, "active", True)
                or not getattr(adv_vehicle, "physical_active", True)
            ):
                continue
            adv_pos = np.asarray(adv_vehicle.position, dtype=float)
            distance = float(np.linalg.norm(ego_pos - adv_pos))
            min_distance = min(min_distance, distance)
        if min_distance < self.safe_distance:
            raw_risk = self.alpha * (1.0 - min_distance / self.safe_distance)

        tau, tau_info = self._adaptive_risk_tau(scenario, ego)
        violation = max(0.0, raw_risk - tau)
        penalty = self.risk_violation_weight * violation
        return float(penalty), {
            "ego_risk_value": float(raw_risk),
            "ego_risk_raw": float(raw_risk),
            "ego_adaptive_tau": float(tau),
            "ego_risk_violation": float(violation),
            "ego_risk_constraint_penalty": float(penalty),
            "ego_min_adv_distance": min_distance if min_distance != float("inf") else -1.0,
            **tau_info,
        }

    def _adaptive_risk_tau(self, scenario, ego):
        modes = [
            RiskFieldReward._interaction_mode(ego, adv)
            for adv in scenario.adv_cav_vehicles
            if getattr(adv, "active", True) and getattr(adv, "physical_active", True)
        ]
        entropy = RiskFieldReward._normalized_entropy(modes)
        coverage = float(len(set(modes)) / max(len(modes), 1)) if modes else 0.0
        speed_ratio = float(np.clip(ego.speed / max(self.max_speed, 1e-6), 0.0, 1.0))
        route_completion = float(np.clip(getattr(scenario, "route_completion", 0.0), 0.0, 1.0))
        ego_performance = float(np.clip(0.5 * speed_ratio + 0.5 * route_completion, 0.0, 1.0))
        performance_easing = 1.0 - ego_performance
        tau = (
            self.base_risk_tau
            + self.tau_entropy_weight * entropy
            + self.tau_coverage_weight * coverage
            + self.tau_performance_weight * performance_easing
        )
        tau = float(np.clip(tau, 0.0, 1.0))
        return tau, {
            "ego_interaction_entropy": float(entropy),
            "ego_interaction_coverage": float(coverage),
            "ego_performance_easing": float(performance_easing),
        }

    def _lane_safety_shaping(self, scenario, ego, previous_lane_index):
        self._steps_since_lane_change += 1
        front_gap = self._front_gap(scenario, ego, ego.lane_id)
        penalty = 0.0
        if front_gap < self.blocked_front_distance:
            penalty = self.front_gap_penalty * (1.0 - front_gap / max(self.blocked_front_distance, 1e-6))

        bonus = 0.0
        lane_change_cost = 0.0
        oscillation_cost = 0.0
        changed_lane = previous_lane_index is not None and int(ego.lane_id) != int(previous_lane_index)
        if changed_lane:
            old_steps = self._steps_since_lane_change
            self._steps_since_lane_change = 0
            lane_change_cost = self.lane_change_cost
            new_front_gap = self._front_gap(scenario, ego, ego.lane_id)
            old_lane_id = int(previous_lane_index)
            old_front_gap = self._front_gap(scenario, ego, old_lane_id)
            front_was_blocked = old_front_gap < self.blocked_front_distance
            if front_was_blocked and new_front_gap > old_front_gap + 5.0:
                bonus = self.lane_change_bonus
            if old_steps < self.oscillation_window:
                oscillation_cost = self.oscillation_penalty * (1.0 - old_steps / max(self.oscillation_window, 1))

        shaping = bonus - penalty - lane_change_cost - oscillation_cost
        return float(shaping), {
            "ego_front_gap": float(front_gap),
            "ego_front_gap_penalty": float(-penalty),
            "ego_safe_lane_bonus": float(bonus),
            "ego_lane_change_cost": float(-lane_change_cost),
            "ego_oscillation_penalty": float(-oscillation_cost),
        }

    def _obstacle_shaping(self, ego, previous_lane_index):
        risk = float(getattr(ego, "obstacle_risk", 0.0))
        distance = float(getattr(ego, "obstacle_distance", -1.0))
        lateral = float(getattr(ego, "obstacle_lateral", 0.0))
        hazardous_lane_change = bool(getattr(ego, "hazardous_lane_change", False))
        hazardous_risk = float(getattr(ego, "hazardous_lane_change_risk", 0.0))
        hazardous_penalty = 0.0
        if hazardous_lane_change:
            hazardous_penalty = self.hazardous_lane_change_penalty * max(
                0.0,
                hazardous_risk - self.hazardous_lane_change_risk_threshold,
            )
        takeover_active = bool(getattr(ego, "obstacle_takeover_active", False))
        takeover_released = bool(getattr(ego, "obstacle_takeover_released", False))
        takeover_penalty = self.obstacle_takeover_penalty if takeover_active else 0.0
        penalty = self.obstacle_penalty_weight * max(risk, 0.0) + hazardous_penalty + takeover_penalty
        self._obstacle_recovery_cooldown_steps += 1
        obstacle_low_speed = (
            risk >= self.obstacle_recovery_risk_threshold
            and distance >= 0.0
            and float(ego.speed) <= self.obstacle_recovery_low_speed
        )
        if obstacle_low_speed:
            self._obstacle_low_speed_pending = True
        changed_lane = previous_lane_index is not None and int(ego.lane_id) != int(previous_lane_index)
        escaped = (
            changed_lane
            and self._last_obstacle_risk >= self.obstacle_escape_threshold
            and risk <= max(0.0, self._last_obstacle_risk - self.obstacle_escape_margin)
            and not ego.crashed
        )
        escape_bonus = self.obstacle_escape_bonus if escaped else 0.0
        recovered = (
            self._obstacle_low_speed_pending
            and self._obstacle_recovery_cooldown_steps >= self.obstacle_recovery_cooldown
            and risk <= self.obstacle_recovery_clear_risk
            and float(ego.speed) >= self.obstacle_recovery_speed
            and not ego.crashed
            and getattr(ego, "on_road", True)
        )
        recovery_bonus = self.obstacle_recovery_bonus if recovered else 0.0
        takeover_release_bonus = (
            self.obstacle_takeover_release_bonus
            if takeover_released and risk <= self.obstacle_recovery_clear_risk and not ego.crashed
            else 0.0
        )
        if recovered:
            self._obstacle_low_speed_pending = False
            self._obstacle_recovery_cooldown_steps = 0
        bonus = escape_bonus + recovery_bonus + takeover_release_bonus
        self._last_obstacle_risk = risk
        return float(penalty), float(bonus), {
            "ego_obstacle_distance": distance,
            "ego_obstacle_lateral": lateral,
            "ego_obstacle_risk": risk,
            "ego_obstacle_escape": bool(escaped),
            "ego_obstacle_recovered": bool(recovered),
            "ego_obstacle_low_speed_pending": bool(self._obstacle_low_speed_pending),
            "ego_obstacle_escape_bonus_value": float(escape_bonus),
            "ego_obstacle_recovery_bonus_value": float(recovery_bonus),
            "ego_obstacle_takeover_penalty_value": float(-takeover_penalty),
            "ego_obstacle_takeover_release_bonus_value": float(takeover_release_bonus),
            "ego_hazardous_lane_change": bool(hazardous_lane_change),
            "ego_hazardous_lane_change_risk": float(hazardous_risk),
            "ego_hazardous_lane_change_penalty": float(-hazardous_penalty),
            "ego_lane_change_cancelled_for_obstacle": bool(getattr(ego, "lane_change_cancelled_for_obstacle", False)),
        }

    @staticmethod
    def _front_gap(scenario, ego, lane_id: int):
        gaps = [
            longitudinal_gap(ego, other)
            for other in scenario.vehicles
            if (
                other is not ego
                and getattr(other, "physical_active", True)
                and other.lane_id == lane_id
                and int(other.road_id) == int(ego.road_id)
                and int(other.section_id) == int(ego.section_id)
                and longitudinal_gap(ego, other) >= 0.0
            )
        ]
        return float(min(gaps) if gaps else 100.0)


class RewardManager:
    """Computes role-specific rewards for the co-evolution environment."""

    def __init__(self, config=None):
        config = config or {}
        self.risk_field_reward = RiskFieldReward()
        self.ego_reward = CatSafetyEgoReward(
            obstacle_penalty_weight=config.get("ego_obstacle_penalty_weight", 2.0),
            obstacle_escape_bonus=config.get("ego_obstacle_escape_bonus", 0.5),
            obstacle_escape_threshold=config.get("ego_obstacle_escape_threshold", 0.35),
            obstacle_escape_margin=config.get("ego_obstacle_escape_margin", 0.15),
            obstacle_recovery_bonus=config.get("ego_obstacle_recovery_bonus", 0.3),
            obstacle_recovery_speed=config.get("ego_obstacle_recovery_speed", 6.0),
            obstacle_recovery_low_speed=config.get("ego_obstacle_recovery_low_speed", 2.5),
            obstacle_recovery_risk_threshold=config.get("ego_obstacle_recovery_risk_threshold", 0.25),
            obstacle_recovery_clear_risk=config.get("ego_obstacle_recovery_clear_risk", 0.10),
            obstacle_recovery_cooldown=config.get("ego_obstacle_recovery_cooldown", 50),
            hazardous_lane_change_penalty=config.get("ego_hazardous_lane_change_penalty", 1.0),
            hazardous_lane_change_risk_threshold=config.get("ego_hazardous_lane_change_risk_threshold", 0.25),
            obstacle_takeover_penalty=config.get("ego_obstacle_takeover_penalty", 0.02),
            obstacle_takeover_release_bonus=config.get("ego_obstacle_takeover_release_bonus", 0.2),
            crash_vehicle_penalty=config.get("ego_crash_vehicle_penalty", 20.0),
            crash_object_penalty=config.get("ego_crash_object_penalty", 20.0),
        )

    def reset(self):
        self.risk_field_reward.reset()
        self.ego_reward.reset()

    def compute(self, scenario, action) -> RewardResult:
        ego_reward, ego_info = self.ego_reward.compute(scenario)
        agents_rewards, risk_fields, risk_deltas, adv_info = self.risk_field_reward.compute_agent_rewards(scenario)
        hdv_rewards = tuple(self._hdv_reward(scenario, vehicle) for vehicle in scenario.hdv_vehicles)
        if agents_rewards:
            global_reward = float(np.mean(agents_rewards))
        else:
            global_reward = ego_reward
        return RewardResult(
            reward=global_reward,
            info={
                "agents_rewards": agents_rewards,
                "regional_rewards": agents_rewards,
                "risk_fields": risk_fields,
                "risk_deltas": risk_deltas,
                **adv_info,
                "ego_reward": ego_reward,
                **ego_info,
                "hdv_reward": float(np.mean(hdv_rewards)) if hdv_rewards else 0.0,
                "hdv_rewards": hdv_rewards,
            },
        )

    @staticmethod
    def _hdv_reward(scenario, vehicle) -> float:
        reward = 0.1 * vehicle.speed
        if vehicle.crashed:
            reward -= 20.0
        return float(reward)
