"""Episode metrics reported through environment info dictionaries."""


class EpisodeMetrics:
    """Collects crash, completion, speed, and termination metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_reward = 0.0
        self.steps = 0

    def update(self, scenario, reward: float) -> dict:
        self.steps += 1
        self.total_reward += float(reward)
        active_vehicles = [
            vehicle for vehicle in scenario.vehicles
            if getattr(vehicle, "active", True) and not vehicle.crashed
        ]
        physical_vehicles = [
            vehicle for vehicle in scenario.vehicles if getattr(vehicle, "physical_active", True)
        ]
        vehicle_speed = [float(vehicle.speed) for vehicle in physical_vehicles]
        vehicle_position = [vehicle.position.tolist() for vehicle in physical_vehicles]
        ego = scenario.ego_vehicle
        ego_crash = bool(ego is not None and ego.crashed)
        any_crash = any(bool(vehicle.crashed) for vehicle in scenario.vehicles)
        non_ego_crash = any(
            bool(vehicle.crashed)
            for vehicle in scenario.vehicles
            if vehicle is not scenario.ego_vehicle
        )
        return {
            "crash": ego_crash,
            "crashed": ego_crash,
            "ego_crash": ego_crash,
            "any_crash": any_crash,
            "non_ego_crash": non_ego_crash,
            "crash_rate": float(scenario.crash_rate),
            "route_completion": float(scenario.route_completion),
            "road_completion_rate": float(scenario.route_completion),
            "route_completion_full": float(getattr(scenario, "route_completion_full", scenario.route_completion)),
            "road_completion_rate_full": float(getattr(scenario, "route_completion_full", scenario.route_completion)),
            "route_completion_distance": float(getattr(scenario, "route_completion_distance", scenario.route_length)),
            "route_length": float(getattr(scenario, "route_length", 0.0)),
            "average_speed": float(scenario.average_speed),
            "ego_route_s": float(ego.route_s) if ego is not None and ego.route_s is not None else 0.0,
            "ego_lateral_offset": float(ego.lateral_offset) if ego is not None else 0.0,
            "ego_heading_error": float(ego.heading_error) if ego is not None else 0.0,
            "ego_road_id": int(ego.road_id) if ego is not None else 0,
            "ego_section_id": int(ego.section_id) if ego is not None else 0,
            "ego_raw_lane_id": int(ego.raw_lane_id) if ego is not None else 0,
            "ego_obstacle_distance": float(ego.obstacle_distance) if ego is not None else -1.0,
            "ego_obstacle_lateral": float(ego.obstacle_lateral) if ego is not None else 0.0,
            "ego_obstacle_risk": float(ego.obstacle_risk) if ego is not None else 0.0,
            "episode_length": self.steps,
            "success": bool(scenario.success),
            "timeout": bool(scenario.timeout),
            "vehicle_speed": vehicle_speed,
            "vehicle_position": vehicle_position,
            "agents_dones": tuple(
                bool(
                    not getattr(vehicle, "active", True)
                    or not getattr(vehicle, "physical_active", True)
                    or scenario.success
                )
                for vehicle in scenario.adv_cav_vehicles
            ),
            "newly_collided_adv_indices": tuple(scenario.newly_collided_adv_indices),
            "colliding_adv_indices": tuple(scenario.colliding_adv_indices),
            "recovered_adv_indices": tuple(scenario.recovered_adv_indices),
            "newly_collided_non_ego_count": int(scenario.newly_collided_non_ego_count),
            "adv_collision_event_count": int(scenario.adv_collision_event_count),
            "adv_adv_collision_event_count": int(scenario.adv_adv_collision_event_count),
            "adv_ego_collision_event_count": int(scenario.adv_ego_collision_event_count),
            "adv_other_collision_event_count": int(scenario.adv_other_collision_event_count),
            "adv_recovery_count": int(scenario.adv_recovery_count),
            "retired_non_ego_count": int(scenario.retired_non_ego_count),
            "policy_inactive_non_ego_count": int(scenario.policy_inactive_non_ego_count),
            "physical_accident_vehicle_count": sum(
                int(vehicle.role != "ego" and vehicle.crashed and getattr(vehicle, "physical_active", True))
                for vehicle in scenario.vehicles
            ),
            "active_adv_count": sum(
                int(
                    getattr(vehicle, "active", True)
                    and getattr(vehicle, "physical_active", True)
                    and not vehicle.crashed
                )
                for vehicle in scenario.adv_cav_vehicles
            ),
            "recovering_adv_count": sum(
                int(
                    getattr(vehicle, "active", True)
                    and getattr(vehicle, "physical_active", True)
                    and vehicle.crashed
                )
                for vehicle in scenario.adv_cav_vehicles
            ),
            "agents_info": tuple(
                (float(vehicle.x), float(vehicle.y), float(vehicle.speed))
                for vehicle in scenario.adv_cav_vehicles
            ),
        }
