"""Scenario lifecycle management for migration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .vehicle_state import VehicleState, longitudinal_gap


class RoadModel:
    """Minimal road helper compatible with old HDV reward code."""

    def __init__(self, state: "ScenarioState"):
        self.state = state

    @property
    def vehicles(self) -> List[VehicleState]:
        return self.state.vehicles

    def surrounding_vehicles(self, vehicle: VehicleState) -> Tuple[Optional[VehicleState], Optional[VehicleState]]:
        same_lane = [
            other for other in self.vehicles
            if other is not vehicle
            and other.lane_id == vehicle.lane_id
            and other.road_id == vehicle.road_id
            and other.section_id == vehicle.section_id
            and other.physical_active
        ]
        front = [other for other in same_lane if longitudinal_gap(vehicle, other) > 0.0]
        rear = [other for other in same_lane if longitudinal_gap(vehicle, other) <= 0.0]
        front_vehicle = min(front, key=lambda item: longitudinal_gap(vehicle, item)) if front else None
        rear_vehicle = max(rear, key=lambda item: longitudinal_gap(vehicle, item)) if rear else None
        return front_vehicle, rear_vehicle


@dataclass
class ScenarioState:
    name: str = "merge_smoke"
    ego_vehicle: Optional[VehicleState] = None
    adv_cav_vehicles: List[VehicleState] = field(default_factory=list)
    hdv_vehicles: List[VehicleState] = field(default_factory=list)
    background_vehicles: List[VehicleState] = field(default_factory=list)
    route_length: float = 200.0
    route_completion_distance: float = 200.0
    dt: float = 0.2
    steps: int = 0
    crash: bool = False
    crash_rate: float = 0.0
    route_completion: float = 0.0
    route_completion_full: float = 0.0
    average_speed: float = 0.0
    success: bool = False
    timeout: bool = False
    newly_collided_adv_indices: Tuple[int, ...] = field(default_factory=tuple)
    colliding_adv_indices: Tuple[int, ...] = field(default_factory=tuple)
    recovered_adv_indices: Tuple[int, ...] = field(default_factory=tuple)
    newly_collided_non_ego_count: int = 0
    adv_collision_event_count: int = 0
    adv_adv_collision_event_count: int = 0
    adv_ego_collision_event_count: int = 0
    adv_other_collision_event_count: int = 0
    adv_recovery_count: int = 0
    retired_non_ego_count: int = 0
    policy_inactive_non_ego_count: int = 0

    def __post_init__(self):
        self.road = RoadModel(self)

    @property
    def vehicles(self) -> List[VehicleState]:
        vehicles = []
        vehicles.extend(self.adv_cav_vehicles)
        if self.ego_vehicle is not None:
            vehicles.append(self.ego_vehicle)
        vehicles.extend(self.hdv_vehicles)
        vehicles.extend(self.background_vehicles)
        return vehicles


class ScenarioManager:
    """Owns simulator objects and per-episode scenario state."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = str(config.get("backend", "mock"))
        self.state = ScenarioState()
        self.client = None
        self.world = None
        self.carla = None
        self._original_settings = None
        self._actors = []
        self._sensors = []
        self._autopilot_actor_ids = set()
        self._actor_roles = {}
        self._actor_indices = {}
        self._collision_actor_ids = set()
        self._reported_collision_actor_ids = set()
        self._collision_first_step = {}
        self._collision_last_event_step = {}
        self._collision_partner_ids = {}
        self._pending_collision_pairs = set()
        self._active_collision_pairs = set()
        self._mock_active_collision_pairs = set()
        self._route_start = None
        self._route_end = None
        self._route_start_xy = None
        self._route_vector_xy = None
        self._route_points_xy = None
        self._route_cumulative_s = None
        self._route_segment_headings = None
        self._route_total_length = 0.0
        self._route_origin_s = 0.0
        self._static_obstacles = []
        self._static_obstacles_ready = False
        self._spawn_transforms = []
        self._control_states = {}
        if self.backend == "carla":
            self._connect()

    def _connect(self):
        try:
            import carla
        except ImportError as exc:
            raise RuntimeError(
                "CARLA backend requested, but the 'carla' Python package is not installed. "
                "Install carla==0.9.13 or use backend='mock' for interface tests."
            ) from exc

        self.carla = carla
        host = self.config.get("host", "127.0.0.1")
        port = int(self.config.get("port", 2000))
        timeout = float(self.config.get("timeout", 60.0))
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        town = self.config.get("town")
        self.world = self.client.get_world()
        if town and not self._world_matches_town(self.world, town):
            self.world = self.client.load_world(town)
        self._configure_world()

    @staticmethod
    def _world_matches_town(world, town: str) -> bool:
        try:
            return str(world.get_map().name).split("/")[-1] == str(town)
        except RuntimeError:
            return False

    def _configure_world(self):
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = bool(self.config.get("synchronous_mode", True))
        settings.fixed_delta_seconds = float(self.config.get("fixed_delta_seconds", 0.05))
        if "no_rendering_mode" in self.config:
            settings.no_rendering_mode = bool(self.config.get("no_rendering_mode", False))
        self.world.apply_settings(settings)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        scenario_config = dict(self.config)
        if options:
            scenario_config.update(options)
        if self.backend == "carla":
            return self._reset_runtime(scenario_config, seed)
        return self._reset_mock(scenario_config, seed)

    def step(self, control_intents: Sequence[Dict[str, Any]]):
        if self.backend == "carla":
            return self._step_runtime(control_intents)
        return self._step_mock(control_intents)

    def close(self):
        self._cleanup_runtime()
        if self.world is not None and self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
            self._original_settings = None
        self.client = None
        self.world = None

    def _reset_runtime(self, scenario_config: Dict[str, Any], seed: Optional[int]):
        if self.world is None:
            self._connect()
        self._cleanup_runtime()
        if bool(scenario_config.get("purge_existing_actors_on_reset", False)):
            self._purge_existing_runtime_actors()
        self._collision_actor_ids = set()
        self._reported_collision_actor_ids = set()
        self._collision_first_step = {}
        self._collision_last_event_step = {}
        self._collision_partner_ids = {}
        self._pending_collision_pairs = set()
        self._active_collision_pairs = set()
        self._mock_active_collision_pairs = set()
        self._spawn_transforms = self._select_spawn_transforms(scenario_config, seed)
        route_length = self._compute_route_length(scenario_config)
        fixed_delta = float(scenario_config.get("fixed_delta_seconds", self.config.get("fixed_delta_seconds", 0.05)))

        num_cav = int(scenario_config.get("num_cav", 1))
        num_hdv = int(scenario_config.get("num_hdv", 3))
        num_background = int(scenario_config.get("num_background", 0))

        adv_actors = []
        ego_actor = None
        hdv_actors = []
        background_actors = []

        spawn_idx = 0
        for idx in range(num_cav):
            actor = self._spawn_vehicle("adv_cav", idx, self._spawn_transforms[spawn_idx])
            adv_actors.append(actor)
            spawn_idx += 1

        ego_actor = self._spawn_vehicle("ego", 0, self._spawn_transforms[spawn_idx])
        spawn_idx += 1

        for idx in range(num_hdv):
            actor = self._spawn_vehicle("hdv", idx, self._spawn_transforms[spawn_idx])
            hdv_actors.append(actor)
            spawn_idx += 1

        for idx in range(num_background):
            actor = self._spawn_vehicle("background", idx, self._spawn_transforms[spawn_idx])
            background_actors.append(actor)
            spawn_idx += 1

        self._configure_traffic_manager_control(scenario_config, hdv_actors)

        if bool(scenario_config.get("tick_after_spawn", True)):
            self._tick_runtime("after_spawn")

        self._set_route_reference(ego_actor, route_length)
        route_length = self._route_total_length or route_length

        self.state = ScenarioState(
            name=scenario_config.get("route_name", "merge_smoke"),
            ego_vehicle=self._vehicle_state_from_actor(ego_actor),
            adv_cav_vehicles=[self._vehicle_state_from_actor(actor) for actor in adv_actors],
            hdv_vehicles=[self._vehicle_state_from_actor(actor) for actor in hdv_actors],
            background_vehicles=[self._vehicle_state_from_actor(actor) for actor in background_actors],
            route_length=route_length,
            dt=fixed_delta,
        )
        self._update_runtime_metrics()
        self._update_spectator(ego_actor)
        return self.state

    def _step_runtime(self, control_intents: Sequence[Dict[str, Any]]):
        default_intent = {"action": 1, "speed_delta": 0.0, "lane_delta": 0, "throttle": 0.35, "brake": 0.0}
        self.state.newly_collided_adv_indices = ()
        self.state.recovered_adv_indices = ()
        self.state.newly_collided_non_ego_count = 0
        for slot, actor in enumerate(self._actors):
            if not self._is_alive(actor):
                continue
            if actor.id in self._autopilot_actor_ids:
                continue
            if actor.id in self._collision_actor_ids and self._actor_roles.get(actor.id) != "ego":
                actor.apply_control(self._collision_recovery_control(actor))
                continue
            intent = control_intents[slot] if slot < len(control_intents) else default_intent
            actor.apply_control(self._to_vehicle_control(actor, intent))
        self._tick_runtime("step")
        self.state.steps += 1
        self._register_new_collisions()
        self._refresh_runtime_collision_states()
        self._sync_runtime_state()
        self._update_runtime_metrics()
        ego_actor = self._actor_by_role("ego")
        self._update_spectator(ego_actor)
        return self.state

    def _select_spawn_transforms(self, scenario_config: Dict[str, Any], seed: Optional[int]):
        spawn_points = list(self.world.get_map().get_spawn_points())
        if not spawn_points:
            raise RuntimeError("The current CARLA map has no spawn points.")
        strategy = str(scenario_config.get("spawn_strategy", "relative_route"))
        if strategy == "relative_route":
            return self._select_relative_route_transforms(scenario_config, seed, spawn_points)
        return self._select_sequential_spawn_transforms(scenario_config, seed, spawn_points)

    def _select_sequential_spawn_transforms(self, scenario_config: Dict[str, Any], seed: Optional[int], spawn_points):
        rng = np.random.RandomState(seed if seed is not None else scenario_config.get("seed", 0))
        required = self._required_spawn_count(scenario_config)
        start_index = int(scenario_config.get("spawn_start_index", 0))
        if bool(scenario_config.get("shuffle_spawn_points", False)):
            rng.shuffle(spawn_points)
        else:
            spawn_points = spawn_points[start_index:] + spawn_points[:start_index]
        if required > len(spawn_points):
            raise RuntimeError("Not enough CARLA spawn points for the requested scenario.")
        return [self._lift_transform(transform) for transform in spawn_points[:required]]

    def _select_relative_route_transforms(self, scenario_config: Dict[str, Any], seed: Optional[int], spawn_points):
        rng = np.random.RandomState(seed if seed is not None else scenario_config.get("seed", 0))
        num_cav = int(scenario_config.get("num_cav", 1))
        num_hdv = int(scenario_config.get("num_hdv", 3))
        num_background = int(scenario_config.get("num_background", 0))
        start_index = int(scenario_config.get("spawn_start_index", 0)) % len(spawn_points)
        if bool(scenario_config.get("randomize_spawn_start_index", False)):
            start_index = int(rng.randint(0, len(spawn_points)))
        ego_transform = self._lift_transform(spawn_points[start_index])
        ego_waypoint = self.world.get_map().get_waypoint(ego_transform.location, project_to_road=True)
        if ego_waypoint is None:
            return self._select_sequential_spawn_transforms(scenario_config, seed, spawn_points)

        adv_offsets = self._offset_list(scenario_config, "adv_offsets", num_cav, default_start=20.0, default_step=20.0)
        adv_lane_offsets = self._int_offset_list(scenario_config, "adv_lane_offsets", num_cav, default_start=0, default_step=0)
        ego_offset = float(scenario_config.get("ego_offset", 0.0))
        ego_lane_offset = int(scenario_config.get("ego_lane_offset", 0))
        hdv_offsets = self._offset_list(scenario_config, "hdv_offsets", num_hdv, default_start=80.0, default_step=20.0)
        hdv_lane_offsets = self._int_offset_list(scenario_config, "hdv_lane_offsets", num_hdv, default_start=0, default_step=0)
        background_offsets = self._offset_list(scenario_config, "background_offsets", num_background, default_start=45.0, default_step=35.0)
        background_lane_offsets = self._int_offset_list(scenario_config, "background_lane_offsets", num_background, default_start=1, default_step=1)
        if bool(scenario_config.get("randomize_scenarios", True)):
            jitter = float(scenario_config.get("relative_offset_jitter", 5.0))
            adv_offsets = self._jitter_offsets(adv_offsets, rng, jitter)
            ego_offset = float(ego_offset + rng.uniform(-jitter, jitter))
            hdv_offsets = self._jitter_offsets(hdv_offsets, rng, jitter)
            background_offsets = self._jitter_offsets(background_offsets, rng, jitter)
            if bool(scenario_config.get("randomize_lane_offsets", False)):
                lane_min = int(scenario_config.get("randomize_lane_offset_min", 0))
                lane_max = int(scenario_config.get("randomize_lane_offset_max", 2))
                ego_lane_offset = self._random_lane_offset(rng, lane_min, lane_max)
                adv_lane_offsets = self._random_lane_offsets(len(adv_lane_offsets), rng, lane_min, lane_max)
                hdv_lane_offsets = self._random_lane_offsets(len(hdv_lane_offsets), rng, lane_min, lane_max)
                background_lane_offsets = self._random_lane_offsets(len(background_lane_offsets), rng, lane_min, lane_max)

        adv_transforms = [
            self._waypoint_transform_at(ego_waypoint, offset, lane_offset=lane_offset)
            for offset, lane_offset in zip(adv_offsets, adv_lane_offsets)
        ]
        ego_transform = self._waypoint_transform_at(ego_waypoint, ego_offset, lane_offset=ego_lane_offset)
        hdv_transforms = [
            self._waypoint_transform_at(ego_waypoint, offset, lane_offset=lane_offset)
            for offset, lane_offset in zip(hdv_offsets, hdv_lane_offsets)
        ]
        background_transforms = [
            self._waypoint_transform_at(ego_waypoint, offset, lane_offset=lane_offset)
            for offset, lane_offset in zip(background_offsets, background_lane_offsets)
        ]

        # Keep action order unchanged: [adv..., ego, hdv..., background...].
        return adv_transforms + [ego_transform] + hdv_transforms + background_transforms

    def _waypoint_transform_at(self, waypoint, offset: float, lane_offset: int = 0):
        offset = float(offset)
        if abs(offset) <= 1e-6:
            target = waypoint
        else:
            candidates = waypoint.next(offset) if offset > 0.0 else waypoint.previous(abs(offset))
            target = candidates[0] if candidates else waypoint
        target = self._shift_waypoint_lane(target, int(lane_offset))
        return self._lift_transform(target.transform)

    def _shift_waypoint_lane(self, waypoint, lane_offset: int):
        target = waypoint
        steps = abs(int(lane_offset))
        if steps == 0:
            return target
        getter = "get_left_lane" if lane_offset > 0 else "get_right_lane"
        for _ in range(steps):
            candidate = getattr(target, getter)()
            if candidate is None or not self._is_driving_lane(candidate):
                break
            target = candidate
        return target

    def _is_driving_lane(self, waypoint) -> bool:
        lane_type = getattr(waypoint, "lane_type", None)
        driving_type = getattr(self.carla.LaneType, "Driving", None)
        return driving_type is None or lane_type == driving_type

    def _offset_list(self, scenario_config: Dict[str, Any], key: str, count: int, default_start: float, default_step: float):
        raw = scenario_config.get(key)
        if raw is None:
            return [default_start + idx * default_step for idx in range(count)]
        values = [float(value) for value in raw]
        if len(values) < count:
            last = values[-1] if values else default_start - default_step
            values.extend(last + default_step * (idx + 1) for idx in range(count - len(values)))
        return values[:count]

    @staticmethod
    def _jitter_offsets(offsets: Sequence[float], rng, jitter: float) -> List[float]:
        jitter = max(float(jitter), 0.0)
        if jitter <= 0.0:
            return [float(offset) for offset in offsets]
        return [float(offset + rng.uniform(-jitter, jitter)) for offset in offsets]

    @staticmethod
    def _random_lane_offset(rng, lane_min: int, lane_max: int) -> int:
        lower = min(int(lane_min), int(lane_max))
        upper = max(int(lane_min), int(lane_max))
        return int(rng.randint(lower, upper + 1))

    def _random_lane_offsets(self, count: int, rng, lane_min: int, lane_max: int) -> List[int]:
        return [self._random_lane_offset(rng, lane_min, lane_max) for _ in range(int(count))]

    def _int_offset_list(self, scenario_config: Dict[str, Any], key: str, count: int, default_start: int, default_step: int):
        raw = scenario_config.get(key)
        if raw is None:
            return [int(default_start + idx * default_step) for idx in range(count)]
        values = [int(value) for value in raw]
        if len(values) < count:
            last = values[-1] if values else default_start - default_step
            values.extend(int(last + default_step * (idx + 1)) for idx in range(count - len(values)))
        return values[:count]

    def _required_spawn_count(self, scenario_config: Dict[str, Any]) -> int:
        return (
            int(scenario_config.get("num_cav", 1))
            + 1
            + int(scenario_config.get("num_hdv", 3))
            + int(scenario_config.get("num_background", 0))
        )

    def _lift_transform(self, transform):
        location = self.carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z + float(self.config.get("spawn_z_offset", 0.5)),
        )
        rotation = self.carla.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw,
            roll=transform.rotation.roll,
        )
        return self.carla.Transform(location, rotation)

    def _spawn_vehicle(self, role: str, index: int, transform):
        blueprints = list(self.world.get_blueprint_library().filter(self.config.get("vehicle_filter", "vehicle.*")))
        four_wheel_blueprints = [bp for bp in blueprints if self._blueprint_int(bp, "number_of_wheels", 4) == 4]
        blueprints = four_wheel_blueprints or blueprints
        if not blueprints:
            raise RuntimeError("No vehicle blueprints found in CARLA blueprint library.")
        blueprint = blueprints[(len(self._actors) + index) % len(blueprints)]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role)
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", colors[(len(self._actors) + index) % len(colors)])
        actor = self._try_spawn_vehicle_with_retries(blueprint, transform)
        if actor is None:
            raise RuntimeError(f"Failed to spawn {role} vehicle at transform {transform}.")
        try:
            actor.set_autopilot(False)
        except RuntimeError:
            pass
        self._actors.append(actor)
        self._actor_roles[actor.id] = role
        self._actor_indices[actor.id] = index
        self._attach_collision_sensor(actor)
        return actor

    def _configure_traffic_manager_control(self, scenario_config: Dict[str, Any], hdv_actors):
        mode = str(scenario_config.get("hdv_control_mode", "")).lower()
        if mode not in {"traffic_manager", "autopilot"} or not hdv_actors:
            return
        try:
            tm_port = int(scenario_config.get("traffic_manager_port", scenario_config.get("tm_port", 8000)))
            traffic_manager = self.client.get_trafficmanager(tm_port)
        except RuntimeError:
            return
        try:
            traffic_manager.set_synchronous_mode(bool(scenario_config.get("synchronous_mode", True)))
        except RuntimeError:
            pass
        global_distance = scenario_config.get("hdv_tm_global_distance_to_leading_vehicle", None)
        if global_distance is not None:
            try:
                traffic_manager.set_global_distance_to_leading_vehicle(float(global_distance))
            except RuntimeError:
                pass
        speed_difference = float(scenario_config.get("hdv_tm_speed_difference", 0.0))
        distance_to_lead = scenario_config.get("hdv_tm_distance_to_leading_vehicle", None)
        auto_lane_change = bool(scenario_config.get("hdv_tm_auto_lane_change", True))
        for actor in hdv_actors:
            try:
                actor.set_autopilot(True, tm_port)
                self._autopilot_actor_ids.add(actor.id)
            except RuntimeError:
                continue
            try:
                traffic_manager.vehicle_percentage_speed_difference(actor, speed_difference)
            except RuntimeError:
                pass
            try:
                traffic_manager.auto_lane_change(actor, auto_lane_change)
            except RuntimeError:
                pass
            if distance_to_lead is not None:
                try:
                    traffic_manager.distance_to_leading_vehicle(actor, float(distance_to_lead))
                except RuntimeError:
                    pass

    def _try_spawn_vehicle_with_retries(self, blueprint, transform):
        offsets = [0.0]
        max_retries = int(self.config.get("spawn_retry_offsets", 6))
        retry_step = float(self.config.get("spawn_retry_step", 1.5))
        for idx in range(1, max(max_retries, 1) + 1):
            direction = 1.0 if idx % 2 else -1.0
            offsets.append(direction * retry_step * float((idx + 1) // 2))
        z_offsets = [0.0, 0.25, 0.5]
        for longitudinal_offset in offsets:
            for z_offset in z_offsets:
                candidate = self._offset_transform(transform, longitudinal_offset, z_offset)
                actor = self.world.try_spawn_actor(blueprint, candidate)
                if actor is not None:
                    return actor
        return None

    def _offset_transform(self, transform, longitudinal_offset: float, z_offset: float):
        yaw = math.radians(float(transform.rotation.yaw))
        location = self.carla.Location(
            x=float(transform.location.x) + float(longitudinal_offset) * math.cos(yaw),
            y=float(transform.location.y) + float(longitudinal_offset) * math.sin(yaw),
            z=float(transform.location.z) + float(z_offset),
        )
        rotation = self.carla.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw,
            roll=transform.rotation.roll,
        )
        return self.carla.Transform(location, rotation)

    def _attach_collision_sensor(self, actor):
        blueprint = self.world.get_blueprint_library().find("sensor.other.collision")
        sensor = self.world.spawn_actor(blueprint, self.carla.Transform(), attach_to=actor)
        sensor.listen(lambda event, actor_id=actor.id: self._on_collision(actor_id, event))
        self._sensors.append(sensor)

    def _on_collision(self, actor_id: int, event):
        self._collision_actor_ids.add(actor_id)
        step = int(self.state.steps)
        self._collision_first_step.setdefault(actor_id, step)
        self._collision_last_event_step[actor_id] = step
        other_actor = getattr(event, "other_actor", None)
        other_id = int(getattr(other_actor, "id", -1))
        if other_id >= 0:
            self._collision_partner_ids[actor_id] = other_id
            pair = tuple(sorted((int(actor_id), other_id)))
        else:
            pair = (int(actor_id), -1)
        self._pending_collision_pairs.add(pair)

    def _register_new_collisions(self):
        pending_pairs = self._pending_collision_pairs
        self._pending_collision_pairs = set()
        new_pairs = pending_pairs - self._active_collision_pairs
        self._active_collision_pairs.update(pending_pairs)

        newly_adv_indices = set()
        newly_non_ego_ids = set()
        for pair in new_pairs:
            actor_ids = [actor_id for actor_id in pair if actor_id >= 0]
            roles = [self._actor_roles.get(actor_id, "other") for actor_id in actor_ids]
            adv_ids = [
                actor_id for actor_id, role in zip(actor_ids, roles) if role == "adv_cav"
            ]
            for actor_id, role in zip(actor_ids, roles):
                if role != "ego":
                    newly_non_ego_ids.add(actor_id)
            for actor_id in adv_ids:
                newly_adv_indices.add(int(self._actor_indices.get(actor_id, 0)))
            if adv_ids:
                self.state.adv_collision_event_count += 1
                if roles.count("adv_cav") >= 2:
                    self.state.adv_adv_collision_event_count += 1
                elif "ego" in roles:
                    self.state.adv_ego_collision_event_count += 1
                else:
                    self.state.adv_other_collision_event_count += 1

        self._reported_collision_actor_ids.update(
            actor_id for pair in new_pairs for actor_id in pair if actor_id >= 0
        )
        self.state.newly_collided_adv_indices = tuple(sorted(newly_adv_indices))
        self.state.newly_collided_non_ego_count = int(len(newly_non_ego_ids))

    def _refresh_runtime_collision_states(self):
        clear_steps = max(int(self.config.get("collision_contact_clear_steps", 8)), 1)
        recovered = []
        for actor_id in list(self._collision_actor_ids):
            if self._actor_roles.get(actor_id) == "ego":
                continue
            last_event = int(self._collision_last_event_step.get(actor_id, self.state.steps))
            if self.state.steps - last_event < clear_steps:
                continue
            if not self._collision_actor_separated(actor_id):
                continue
            self._collision_actor_ids.discard(actor_id)
            self._reported_collision_actor_ids.discard(actor_id)
            self._collision_first_step.pop(actor_id, None)
            self._collision_last_event_step.pop(actor_id, None)
            self._collision_partner_ids.pop(actor_id, None)
            if self._actor_roles.get(actor_id) == "adv_cav":
                recovered.append(int(self._actor_indices.get(actor_id, 0)))

        active_ids = set(self._collision_actor_ids)
        self._active_collision_pairs = {
            pair for pair in self._active_collision_pairs
            if any(actor_id in active_ids for actor_id in pair if actor_id >= 0)
        }
        self.state.recovered_adv_indices = tuple(sorted(set(recovered)))
        self.state.adv_recovery_count += len(set(recovered))
        self.state.colliding_adv_indices = tuple(sorted(
            int(self._actor_indices.get(actor_id, 0))
            for actor_id in self._collision_actor_ids
            if self._actor_roles.get(actor_id) == "adv_cav"
        ))

    def _collision_actor_separated(self, actor_id):
        actor = self._actor_by_id(actor_id)
        partner = self._actor_by_id(self._collision_partner_ids.get(actor_id))
        if actor is None or partner is None:
            return True
        actor_location = actor.get_location()
        partner_location = partner.get_location()
        distance = math.sqrt(
            (actor_location.x - partner_location.x) ** 2
            + (actor_location.y - partner_location.y) ** 2
        )
        clearance = float(self.config.get("collision_recovery_clearance", 6.0))
        return distance >= clearance

    def _collision_recovery_control(self, actor):
        first_step = int(self._collision_first_step.get(actor.id, self.state.steps))
        elapsed = max(int(self.state.steps) - first_step, 0)
        brake_steps = max(int(self.config.get("collision_recovery_brake_steps", 3)), 0)
        if elapsed < brake_steps and self._actor_speed(actor) > 1.0:
            return self.carla.VehicleControl(throttle=0.0, brake=0.65, steer=0.0)

        partner = self._actor_by_id(self._collision_partner_ids.get(actor.id))
        reverse = False
        throttle = float(self.config.get("collision_recovery_throttle", 0.35))
        if partner is not None and self._actor_route_gap(actor, partner) >= 0.0:
            reverse = True
            throttle = float(self.config.get("collision_recovery_reverse_throttle", 0.25))
        steer_sign = -1 if int(self._actor_indices.get(actor.id, 0)) % 2 == 0 else 1
        waypoint = self.world.get_map().get_waypoint(actor.get_location(), project_to_road=True)
        if waypoint is not None and self._adjacent_driving_waypoint(waypoint, steer_sign) is None:
            opposite = -steer_sign
            steer_sign = opposite if self._adjacent_driving_waypoint(waypoint, opposite) is not None else 0
        steer = steer_sign * float(self.config.get("collision_recovery_steer", 0.18))
        return self.carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            brake=0.0,
            steer=float(np.clip(steer, -1.0, 1.0)),
            reverse=bool(reverse),
        )

    def _actor_by_id(self, actor_id):
        if actor_id is None:
            return None
        for actor in self._actors:
            if actor.id == actor_id and self._is_alive(actor):
                return actor
        return None

    def _actor_by_role(self, role: str):
        for actor in self._actors:
            if self._is_alive(actor) and self._actor_roles.get(actor.id) == role:
                return actor
        return None

    def _update_spectator(self, target_actor):
        if not bool(self.config.get("spectator_follow", True)):
            return
        if target_actor is None or not self._is_alive(target_actor):
            return
        spectator = self.world.get_spectator()
        transform = target_actor.get_transform()
        yaw = math.radians(transform.rotation.yaw)
        distance = float(self.config.get("spectator_distance", 18.0))
        height = float(self.config.get("spectator_height", 8.0))
        pitch = float(self.config.get("spectator_pitch", -25.0))
        location = self.carla.Location(
            x=transform.location.x - distance * math.cos(yaw),
            y=transform.location.y - distance * math.sin(yaw),
            z=transform.location.z + height,
        )
        rotation = self.carla.Rotation(pitch=pitch, yaw=transform.rotation.yaw, roll=0.0)
        spectator.set_transform(self.carla.Transform(location, rotation))

    def _to_vehicle_control(self, actor, intent: Dict[str, Any]):
        if not bool(self.config.get("use_high_level_intent", True)):
            return self._direct_vehicle_control(intent)

        state = self._control_state(actor)
        if state.get("lane_change_cooldown", 0) > 0:
            state["lane_change_cooldown"] -= 1
        current_speed = self._actor_speed(actor)
        target_speed_min = float(self.config.get("target_speed_min", 2.0))
        target_speed_max = float(self.config.get("target_speed_max", 28.0))
        cruise_speed = float(self.config.get("target_speed_cruise", 18.0))
        if state["target_speed"] is None:
            state["target_speed"] = max(current_speed, cruise_speed)

        state["front_gap"] = -1.0
        state["front_speed"] = -1.0
        state["safety_speed_limit"] = -1.0
        state["safety_brake"] = 0.0
        state["obstacle_distance"] = -1.0
        state["obstacle_risk"] = 0.0
        state["obstacle_label"] = ""
        state["obstacle_lane_lateral"] = 0.0
        state["obstacle_escape_available"] = False
        state["obstacle_avoidance_active"] = False
        state["obstacle_avoidance_blocked_reason"] = ""
        state["obstacle_speed_limited"] = False
        state["obstacle_stop_active"] = False
        state["obstacle_takeover_released"] = False
        state["clear_path_recovery_active"] = False
        state["target_lane_obstacle_risk"] = 0.0
        state["lane_obstacle_veto"] = False
        state["lane_change_cancelled_for_obstacle"] = False
        state["hazardous_lane_change"] = False
        state["hazardous_lane_change_risk"] = 0.0
        state["route_takeover_active"] = False
        state["route_takeover_reason"] = ""
        state["route_current_error_deg"] = 0.0
        state["route_target_error_deg"] = 0.0

        speed_delta = float(intent.get("speed_delta", 0.0))
        if abs(speed_delta) > 1e-6:
            state["target_speed"] = float(np.clip(state["target_speed"] + speed_delta, target_speed_min, target_speed_max))
        else:
            recovery = float(self.config.get("target_speed_recovery", 0.12))
            if recovery > 0.0 and state["target_speed"] < cruise_speed:
                state["target_speed"] = float(np.clip(
                    state["target_speed"] + recovery * (cruise_speed - state["target_speed"]),
                    target_speed_min,
                    target_speed_max,
                ))

        waypoint = self.world.get_map().get_waypoint(actor.get_location(), project_to_road=True)
        lane_delta = int(intent.get("lane_delta", 0))
        can_start_lane_change = state.get("lane_change_intent") is None and state.get("lane_change_cooldown", 0) <= 0
        obstacle_query = self._lane_obstacle_query_for_actor(actor, waypoint)
        current_obstacle_distance = float(obstacle_query.get("distance", -1.0))
        current_obstacle_risk = float(obstacle_query.get("risk", 0.0))
        state["obstacle_distance"] = float(current_obstacle_distance)
        state["obstacle_risk"] = float(current_obstacle_risk)
        state["obstacle_label"] = str(obstacle_query.get("label", ""))
        state["obstacle_lane_lateral"] = float(obstacle_query.get("lane_lateral", 0.0))
        if (
            waypoint is not None
            and state.get("lane_change_intent") is not None
            and state.get("target_lane_id") is not None
            and int(waypoint.lane_id) == int(state.get("target_lane_id"))
        ):
            state["lane_change_success"] = True
            state["lane_change_intent"] = None
            state["lane_change_timer"] = 0
            if current_obstacle_risk >= float(self.config.get("ego_obstacle_safety_risk_threshold", 0.25)):
                state["lane_change_cooldown"] = 0
            else:
                state["lane_change_cooldown"] = int(self.config.get("lane_change_cooldown_steps", 20))
            can_start_lane_change = state.get("lane_change_cooldown", 0) <= 0
        if state.get("lane_change_intent") is not None:
            cancelled = self._cancel_obstacle_hazard_lane_change(
                actor,
                waypoint,
                state,
                current_obstacle_risk,
                current_obstacle_distance,
            )
            if cancelled:
                lane_delta = 0
                can_start_lane_change = state.get("lane_change_intent") is None and state.get("lane_change_cooldown", 0) <= 0
        route_option = self._select_route_correction_lane(actor, waypoint, current_obstacle_risk)
        if route_option is not None:
            target_waypoint = route_option["target_waypoint"]
            route_lane_delta = int(route_option["lane_delta"])
            state["route_takeover_active"] = True
            state["route_takeover_reason"] = str(route_option["reason"])
            state["route_current_error_deg"] = float(route_option["current_error_deg"])
            state["route_target_error_deg"] = float(route_option["target_error_deg"])
            state["lane_change_cooldown"] = 0
            state["target_lane_id"] = target_waypoint.lane_id
            state["lane_change_intent"] = "right" if route_lane_delta > 0 else "left"
            state["lane_change_timer"] = 0
            lane_delta = 0
            can_start_lane_change = False
        escape_option = self._select_obstacle_escape_lane(actor, waypoint, current_obstacle_risk)
        state["obstacle_escape_available"] = escape_option is not None
        takeover_active = self._update_obstacle_takeover_state(
            actor,
            state,
            current_obstacle_risk,
            current_obstacle_distance,
            escape_option,
        )
        if takeover_active:
            lane_delta = 0
            takeover_target_lane_id = state.get("obstacle_takeover_target_lane_id")
            if takeover_target_lane_id is not None:
                state["target_lane_id"] = takeover_target_lane_id
        if escape_option is not None:
            obstacle_override = takeover_active or self._can_override_for_obstacle_avoidance(state, current_obstacle_distance)
            if obstacle_override:
                lane_delta = int(escape_option["lane_delta"])
                state["lane_change_cooldown"] = 0
                state["obstacle_avoidance_active"] = True
                target_waypoint = escape_option.get("target_waypoint")
                if target_waypoint is not None:
                    state["obstacle_takeover_target_lane_id"] = target_waypoint.lane_id
                    state["obstacle_takeover_lane_delta"] = int(lane_delta)
                    state["target_lane_id"] = target_waypoint.lane_id
                    state["lane_change_intent"] = "right" if lane_delta > 0 else "left"
                    state["lane_change_timer"] = 0
                    lane_delta = 0
                can_start_lane_change = True
            elif state.get("lane_change_intent") is not None:
                state["obstacle_avoidance_blocked_reason"] = "lane_change_in_progress"
            elif state.get("lane_change_cooldown", 0) > 0:
                state["obstacle_avoidance_blocked_reason"] = "cooldown"
            else:
                state["obstacle_avoidance_blocked_reason"] = "policy"
        elif takeover_active:
            state["obstacle_avoidance_blocked_reason"] = "takeover_no_escape"
        if waypoint is not None and lane_delta != 0 and can_start_lane_change:
            target_waypoint = self._adjacent_driving_waypoint(waypoint, lane_delta)
            target_obstacle_risk = self._lane_obstacle_risk_for_waypoint(actor, target_waypoint)
            state["target_lane_obstacle_risk"] = float(target_obstacle_risk)
            obstacle_safe = self._target_lane_obstacle_safe(current_obstacle_risk, target_obstacle_risk)
            route_aligned = self._waypoint_route_aligned(target_waypoint)
            if target_waypoint is not None and route_aligned and obstacle_safe and self._lane_change_safe(actor, target_waypoint, lane_delta):
                state["target_lane_id"] = target_waypoint.lane_id
                state["lane_change_intent"] = "right" if lane_delta > 0 else "left"
                state["lane_change_timer"] = 0
                state["lane_change_success"] = False
                state["lane_change_failed"] = False
            else:
                state["lane_change_failed"] = True
                state["lane_obstacle_veto"] = bool(target_waypoint is not None and (not obstacle_safe or not route_aligned))
                state["hazardous_lane_change"] = bool(target_waypoint is not None and (not obstacle_safe or not route_aligned))
                state["hazardous_lane_change_risk"] = float(target_obstacle_risk)
                state["lane_change_cooldown"] = int(self.config.get("failed_lane_change_cooldown", 10))

        if waypoint is not None and state.get("target_lane_id") is None:
            state["target_lane_id"] = waypoint.lane_id

        self._apply_front_safety_limit(actor, waypoint, state, target_speed_min)
        self._apply_obstacle_safety_limit(actor, state, target_speed_min)
        self._apply_clear_path_low_speed_recovery(actor, state)

        takeover_waypoint = self._obstacle_takeover_waypoint(actor, waypoint, state)
        target_waypoint = takeover_waypoint or self._target_lane_waypoint(actor, waypoint, state.get("target_lane_id"))
        steer = self._lateral_control(actor, target_waypoint) if target_waypoint is not None else float(intent.get("steer", 0.0))
        if takeover_waypoint is not None:
            steer = float(np.clip(
                steer * float(self.config.get("ego_obstacle_takeover_steer_gain", 1.35)),
                -float(self.config.get("ego_obstacle_takeover_max_steer", 0.65)),
                float(self.config.get("ego_obstacle_takeover_max_steer", 0.65)),
            ))
        throttle, brake = self._longitudinal_control(current_speed, state["target_speed"])
        if state.get("safety_brake", 0.0) > 0.0:
            brake = max(brake, float(state["safety_brake"]))
            throttle = 0.0

        if state.get("lane_change_intent") is not None:
            state["lane_change_timer"] += 1
            if waypoint is not None and waypoint.lane_id == state.get("target_lane_id"):
                state["lane_change_success"] = True
                state["lane_change_intent"] = None
                state["lane_change_cooldown"] = int(self.config.get("lane_change_cooldown_steps", 20))
            elif state["lane_change_timer"] > int(self.config.get("max_lane_change_steps", 25)):
                state["lane_change_failed"] = True
                state["lane_change_intent"] = None
                state["lane_change_cooldown"] = int(self.config.get("failed_lane_change_cooldown", 10))
                if waypoint is not None:
                    state["target_lane_id"] = waypoint.lane_id

        state["previous_action"] = int(intent.get("action", state.get("previous_action", 1)))
        return self.carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0)),
            steer=float(np.clip(steer, -1.0, 1.0)),
        )

    def _direct_vehicle_control(self, intent: Dict[str, Any]):
        lane_change = int(intent.get("lane_change", 0))
        steer = float(intent.get("steer", 0.0))
        if lane_change < 0:
            steer = -0.35
        elif lane_change > 0:
            steer = 0.35
        return self.carla.VehicleControl(
            throttle=float(np.clip(intent.get("throttle", 0.0), 0.0, 1.0)),
            brake=float(np.clip(intent.get("brake", 0.0), 0.0, 1.0)),
            steer=float(np.clip(steer, -1.0, 1.0)),
        )

    def _control_state(self, actor):
        if actor.id not in self._control_states:
            self._control_states[actor.id] = {
                "target_speed": None,
                "target_lane_id": None,
                "lane_change_intent": None,
                "lane_change_timer": 0,
                "lane_change_success": False,
                "lane_change_failed": False,
                "lane_change_cooldown": 0,
                "previous_action": 1,
                "front_gap": -1.0,
                "front_speed": -1.0,
                "safety_speed_limit": -1.0,
                "safety_brake": 0.0,
                "obstacle_distance": -1.0,
                "obstacle_risk": 0.0,
                "obstacle_label": "",
                "obstacle_lane_lateral": 0.0,
                "obstacle_escape_available": False,
                "obstacle_avoidance_active": False,
                "obstacle_avoidance_blocked_reason": "",
                "obstacle_speed_limited": False,
                "obstacle_stop_active": False,
                "obstacle_takeover_active": False,
                "obstacle_takeover_timer": 0,
                "obstacle_takeover_clear_steps": 0,
                "obstacle_takeover_released": False,
                "obstacle_takeover_target_lane_id": None,
                "obstacle_takeover_lane_delta": 0,
                "clear_path_recovery_active": False,
                "target_lane_obstacle_risk": 0.0,
                "lane_obstacle_veto": False,
                "lane_change_cancelled_for_obstacle": False,
                "hazardous_lane_change": False,
                "hazardous_lane_change_risk": 0.0,
                "route_takeover_active": False,
                "route_takeover_reason": "",
                "route_current_error_deg": 0.0,
                "route_target_error_deg": 0.0,
            }
        return self._control_states[actor.id]

    def control_debug_info(self) -> Dict[str, Any]:
        actor = self._actor_by_role("ego")
        if actor is None or actor.id not in self._control_states:
            return {
                "ego_target_speed": -1.0,
                "ego_target_lane_id": -999,
                "ego_lane_change_intent": "",
                "ego_lane_change_timer": 0,
                "ego_lane_change_success": False,
                "ego_lane_change_failed": False,
                "ego_lane_change_cooldown": 0,
                "ego_last_action": -1,
                "ego_front_gap_control": -1.0,
                "ego_front_speed_control": -1.0,
                "ego_safety_speed_limit": -1.0,
                "ego_safety_brake": 0.0,
                "ego_obstacle_distance_control": -1.0,
                "ego_obstacle_risk_control": 0.0,
                "ego_obstacle_label": "",
                "ego_obstacle_lane_lateral": 0.0,
                "ego_obstacle_escape_available": False,
                "ego_obstacle_avoidance_active": False,
                "ego_obstacle_avoidance_blocked_reason": "",
                "ego_obstacle_speed_limited": False,
                "ego_obstacle_stop_active": False,
                "ego_obstacle_takeover_active": False,
                "ego_obstacle_takeover_timer": 0,
                "ego_obstacle_takeover_clear_steps": 0,
                "ego_obstacle_takeover_released": False,
                "ego_obstacle_takeover_target_lane_id": -999,
                "ego_clear_path_recovery_active": False,
                "ego_target_lane_obstacle_risk": 0.0,
                "ego_lane_obstacle_veto": False,
                "ego_lane_change_cancelled_for_obstacle": False,
                "ego_hazardous_lane_change": False,
                "ego_hazardous_lane_change_risk": 0.0,
                "ego_route_takeover_active": False,
                "ego_route_takeover_reason": "",
                "ego_route_current_error_deg": 0.0,
                "ego_route_target_error_deg": 0.0,
                **self._adv_control_debug_info(),
            }
        state = self._control_states[actor.id]
        return {
            "ego_target_speed": float(state.get("target_speed") if state.get("target_speed") is not None else -1.0),
            "ego_target_lane_id": int(state.get("target_lane_id") if state.get("target_lane_id") is not None else -999),
            "ego_lane_change_intent": str(state.get("lane_change_intent") or ""),
            "ego_lane_change_timer": int(state.get("lane_change_timer", 0)),
            "ego_lane_change_success": bool(state.get("lane_change_success", False)),
            "ego_lane_change_failed": bool(state.get("lane_change_failed", False)),
            "ego_lane_change_cooldown": int(state.get("lane_change_cooldown", 0)),
            "ego_last_action": int(state.get("previous_action", -1)),
            "ego_front_gap_control": float(state.get("front_gap", -1.0)),
            "ego_front_speed_control": float(state.get("front_speed", -1.0)),
            "ego_safety_speed_limit": float(state.get("safety_speed_limit", -1.0)),
            "ego_safety_brake": float(state.get("safety_brake", 0.0)),
            "ego_obstacle_distance_control": float(state.get("obstacle_distance", -1.0)),
            "ego_obstacle_risk_control": float(state.get("obstacle_risk", 0.0)),
            "ego_obstacle_label": str(state.get("obstacle_label", "")),
            "ego_obstacle_lane_lateral": float(state.get("obstacle_lane_lateral", 0.0)),
            "ego_obstacle_escape_available": bool(state.get("obstacle_escape_available", False)),
            "ego_obstacle_avoidance_active": bool(state.get("obstacle_avoidance_active", False)),
            "ego_obstacle_avoidance_blocked_reason": str(state.get("obstacle_avoidance_blocked_reason", "")),
            "ego_obstacle_speed_limited": bool(state.get("obstacle_speed_limited", False)),
            "ego_obstacle_stop_active": bool(state.get("obstacle_stop_active", False)),
            "ego_obstacle_takeover_active": bool(state.get("obstacle_takeover_active", False)),
            "ego_obstacle_takeover_timer": int(state.get("obstacle_takeover_timer", 0)),
            "ego_obstacle_takeover_clear_steps": int(state.get("obstacle_takeover_clear_steps", 0)),
            "ego_obstacle_takeover_released": bool(state.get("obstacle_takeover_released", False)),
            "ego_obstacle_takeover_target_lane_id": int(
                state.get("obstacle_takeover_target_lane_id")
                if state.get("obstacle_takeover_target_lane_id") is not None
                else -999
            ),
            "ego_clear_path_recovery_active": bool(state.get("clear_path_recovery_active", False)),
            "ego_target_lane_obstacle_risk": float(state.get("target_lane_obstacle_risk", 0.0)),
            "ego_lane_obstacle_veto": bool(state.get("lane_obstacle_veto", False)),
            "ego_lane_change_cancelled_for_obstacle": bool(state.get("lane_change_cancelled_for_obstacle", False)),
            "ego_hazardous_lane_change": bool(state.get("hazardous_lane_change", False)),
            "ego_hazardous_lane_change_risk": float(state.get("hazardous_lane_change_risk", 0.0)),
            "ego_route_takeover_active": bool(state.get("route_takeover_active", False)),
            "ego_route_takeover_reason": str(state.get("route_takeover_reason", "")),
            "ego_route_current_error_deg": float(state.get("route_current_error_deg", 0.0)),
            "ego_route_target_error_deg": float(state.get("route_target_error_deg", 0.0)),
            **self._adv_control_debug_info(),
        }

    def _adv_control_debug_info(self):
        states = [
            self._control_states[actor.id]
            for actor in self._actors
            if (
                self._is_alive(actor)
                and self._actor_roles.get(actor.id) == "adv_cav"
                and actor.id in self._control_states
                and actor.id not in self._collision_actor_ids
            )
        ]
        valid_gaps = [float(state.get("front_gap", -1.0)) for state in states if state.get("front_gap", -1.0) >= 0.0]
        safety_brakes = [float(state.get("safety_brake", 0.0)) for state in states]
        safety_limits = [
            float(state.get("safety_speed_limit", -1.0))
            for state in states if state.get("safety_speed_limit", -1.0) >= 0.0
        ]
        return {
            "adv_front_safety_active_count": sum(int(value > 0.0) for value in safety_brakes),
            "adv_front_safety_mean_brake": float(np.mean(safety_brakes)) if safety_brakes else 0.0,
            "adv_front_control_min_gap": min(valid_gaps) if valid_gaps else -1.0,
            "adv_front_control_mean_speed_limit": float(np.mean(safety_limits)) if safety_limits else -1.0,
        }

    def _actor_speed(self, actor):
        velocity = actor.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def _apply_front_safety_limit(self, actor, waypoint, state: Dict[str, Any], target_speed_min: float):
        role = self._actor_roles.get(actor.id, "background")
        if role not in {"ego", "adv_cav"} or waypoint is None:
            return
        front_actor, front_gap = self._same_lane_front_actor(actor, waypoint)
        if front_actor is None:
            return

        front_speed = self._actor_speed(front_actor)
        prefix = "ego" if role == "ego" else "adv"
        default_distance = 22.0 if role == "ego" else 14.0
        default_emergency_gap = 8.0 if role == "ego" else 5.0
        default_headway = 1.5 if role == "ego" else 0.8
        default_light_gap = 12.0 if role == "ego" else 8.0
        default_light_brake = 0.20 if role == "ego" else 0.12
        default_emergency_brake = 0.45 if role == "ego" else 0.30
        safety_distance = float(self.config.get(f"{prefix}_front_safety_distance", default_distance))
        emergency_gap = float(self.config.get(f"{prefix}_front_emergency_gap", default_emergency_gap))
        time_headway = max(float(self.config.get(f"{prefix}_front_time_headway", default_headway)), 1e-6)
        state["front_gap"] = float(front_gap)
        state["front_speed"] = float(front_speed)

        if front_gap >= safety_distance:
            return

        safety_limit = front_speed + max(front_gap - emergency_gap, 0.0) / time_headway
        safety_limit = max(float(target_speed_min), safety_limit)
        state["safety_speed_limit"] = float(safety_limit)
        state["target_speed"] = float(min(float(state["target_speed"]), safety_limit))

        light_brake_gap = float(self.config.get(f"{prefix}_front_light_brake_gap", default_light_gap))
        if front_gap <= emergency_gap:
            state["target_speed"] = float(target_speed_min)
            state["safety_brake"] = float(
                self.config.get(f"{prefix}_front_emergency_brake", default_emergency_brake)
            )
        elif front_gap <= light_brake_gap:
            state["safety_brake"] = float(
                self.config.get(f"{prefix}_front_light_brake", default_light_brake)
            )

    def _same_lane_front_actor(self, actor, waypoint):
        lane_id = waypoint.lane_id
        front_actor = None
        front_gap = float("inf")
        for other in self._actors:
            if (
                other.id == actor.id
                or not self._is_alive(other)
            ):
                continue
            other_wp = self.world.get_map().get_waypoint(other.get_location(), project_to_road=True)
            if (
                other_wp is None
                or other_wp.lane_id != lane_id
                or other_wp.road_id != waypoint.road_id
                or other_wp.section_id != waypoint.section_id
            ):
                continue
            gap = self._actor_route_gap(actor, other)
            if 0.0 <= gap < front_gap:
                front_actor = other
                front_gap = gap
        return front_actor, front_gap

    def _apply_obstacle_safety_limit(self, actor, state: Dict[str, Any], target_speed_min: float):
        if self._actor_roles.get(actor.id) != "ego":
            return
        risk = float(state.get("obstacle_risk", 0.0))
        distance = float(state.get("obstacle_distance", -1.0))
        active_threshold = float(self.config.get("ego_obstacle_safety_risk_threshold", 0.25))
        if risk < active_threshold or distance < 0.0:
            return

        lookahead = max(float(self.config.get("ego_obstacle_lookahead_distance", 20.0)), 1e-6)
        emergency_distance = float(self.config.get("ego_obstacle_emergency_distance", 8.0))
        critical_distance = float(self.config.get("ego_obstacle_critical_distance", 3.0))
        stop_distance = float(self.config.get("ego_obstacle_stop_distance", critical_distance))
        min_pass_speed = max(float(self.config.get("ego_obstacle_min_pass_speed", 8.0)), float(target_speed_min))
        escape_available = bool(state.get("obstacle_escape_available", False))
        min_speed = float(target_speed_min)
        if escape_available and distance > emergency_distance:
            return

        safe_speed = min_pass_speed + max(distance - critical_distance, 0.0) / max(
            float(self.config.get("ego_obstacle_time_headway", 2.0)), 1e-6
        )
        if escape_available and distance > critical_distance:
            safe_speed = max(safe_speed, min_pass_speed)
        current_limit = float(state.get("safety_speed_limit", -1.0))
        if current_limit < 0.0:
            current_limit = safe_speed
        else:
            current_limit = min(current_limit, safe_speed)
        state["safety_speed_limit"] = float(max(min_speed, current_limit))
        target_speed = float(min(float(state["target_speed"]), state["safety_speed_limit"]))
        if escape_available and distance > stop_distance:
            escape_floor = float(self.config.get("ego_obstacle_escape_min_speed", min_pass_speed))
            target_speed = min(max(target_speed, escape_floor), float(state["safety_speed_limit"]))
        state["target_speed"] = float(target_speed)
        state["obstacle_speed_limited"] = True
        should_stop = distance <= stop_distance
        if should_stop:
            state["target_speed"] = min_speed
            state["obstacle_stop_active"] = True
            state["safety_brake"] = max(
                float(state.get("safety_brake", 0.0)),
                float(self.config.get("ego_obstacle_emergency_brake", 0.35)),
            )
        elif distance <= emergency_distance:
            state["safety_brake"] = max(
                float(state.get("safety_brake", 0.0)),
                float(self.config.get("ego_obstacle_light_brake", 0.08)) * risk,
            )

    def _apply_clear_path_low_speed_recovery(self, actor, state: Dict[str, Any]):
        if self._actor_roles.get(actor.id) != "ego":
            return
        if float(state.get("safety_brake", 0.0)) > 0.0:
            return
        target_speed = float(state.get("target_speed", 0.0))
        recovery_speed = float(self.config.get("ego_clear_path_recovery_speed", 8.0))
        trigger_speed = float(self.config.get("ego_clear_path_recovery_trigger_speed", 4.0))
        if target_speed > trigger_speed:
            return

        obstacle_risk = float(state.get("obstacle_risk", 0.0))
        clear_obstacle_risk = float(self.config.get("ego_clear_path_obstacle_risk", 0.10))
        if obstacle_risk > clear_obstacle_risk:
            return

        front_gap = float(state.get("front_gap", -1.0))
        clear_front_gap = float(self.config.get("ego_clear_path_front_gap", 25.0))
        if 0.0 <= front_gap < clear_front_gap:
            return

        if state.get("lane_change_intent") is not None and not bool(
            self.config.get("ego_clear_path_recovery_during_lane_change", False)
        ):
            return

        state["target_speed"] = float(max(target_speed, recovery_speed))
        state["clear_path_recovery_active"] = True

    def _select_obstacle_escape_lane(self, actor, waypoint, current_risk: float):
        if self._actor_roles.get(actor.id) != "ego" or waypoint is None:
            return None
        if float(current_risk) < float(self.config.get("ego_obstacle_safety_risk_threshold", 0.25)):
            return None
        distance, _, _ = self._lane_obstacle_risk_for_actor(actor, waypoint)
        if distance < 0.0 or distance > float(self.config.get("ego_obstacle_avoidance_distance", 18.0)):
            return None

        improvement_margin = float(self.config.get("ego_obstacle_escape_risk_margin", 0.15))
        candidates = []
        for lane_delta in (-1, 1):
            target_waypoint = self._adjacent_driving_waypoint(waypoint, lane_delta)
            if target_waypoint is None:
                continue
            if not self._waypoint_route_aligned(target_waypoint):
                continue
            target_risk = self._lane_obstacle_risk_for_waypoint(actor, target_waypoint)
            if target_risk > max(0.0, float(current_risk) - improvement_margin):
                continue
            if not self._target_lane_obstacle_safe(current_risk, target_risk):
                continue
            if not self._lane_change_safe(actor, target_waypoint, lane_delta):
                continue
            candidates.append({
                "lane_delta": int(lane_delta),
                "target_waypoint": target_waypoint,
                "target_risk": float(target_risk),
            })
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item["target_risk"], abs(item["lane_delta"])))

    def _select_route_correction_lane(self, actor, waypoint, current_risk: float):
        if self._actor_roles.get(actor.id) != "ego" or waypoint is None:
            return None
        if not bool(self.config.get("ego_route_takeover_enabled", True)):
            return None

        lookahead = float(self.config.get("ego_route_future_heading_lookahead", 25.0))
        current_error = self._waypoint_route_heading_error_deg(waypoint, lookahead)
        mismatch_threshold = float(self.config.get("ego_route_lane_mismatch_threshold_deg", 15.0))
        improvement_margin = float(self.config.get("ego_route_lane_improvement_margin_deg", 10.0))
        current_lateral = self._actor_route_lateral(actor)
        recovery_lateral = float(self.config.get("ego_route_corridor_recovery_lateral", 8.0))
        route_mismatch = current_error > mismatch_threshold
        corridor_mismatch = abs(current_lateral) > recovery_lateral
        if not route_mismatch and not corridor_mismatch:
            return None

        obstacle_margin = float(self.config.get("ego_route_corridor_obstacle_risk_margin", 0.10))
        candidates = []
        for lane_delta in (-1, 1):
            target_waypoint = self._adjacent_driving_waypoint(waypoint, lane_delta)
            if target_waypoint is None:
                continue
            target_error = self._waypoint_route_heading_error_deg(target_waypoint, lookahead)
            target_lateral = self._waypoint_route_lateral(target_waypoint)
            heading_better = route_mismatch and target_error + improvement_margin < current_error
            lateral_better = corridor_mismatch and abs(target_lateral) + 0.5 < abs(current_lateral)
            if not heading_better and not lateral_better:
                continue
            target_risk = self._lane_obstacle_risk_for_waypoint(actor, target_waypoint)
            if target_risk > float(current_risk) + obstacle_margin:
                continue
            if not self._lane_change_safe(actor, target_waypoint, lane_delta):
                continue
            candidates.append({
                "lane_delta": int(lane_delta),
                "target_waypoint": target_waypoint,
                "current_error_deg": float(current_error),
                "target_error_deg": float(target_error),
                "target_lateral": float(target_lateral),
                "target_risk": float(target_risk),
                "reason": "route_heading" if heading_better else "route_corridor",
            })
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item["target_error_deg"],
                abs(item["target_lateral"]),
                item["target_risk"],
            ),
        )

    def _update_obstacle_takeover_state(
        self,
        actor,
        state: Dict[str, Any],
        current_risk: float,
        current_distance: float,
        escape_option,
    ) -> bool:
        if self._actor_roles.get(actor.id) != "ego":
            return False
        enabled = bool(self.config.get("ego_obstacle_takeover_enabled", True))
        if not enabled:
            state["obstacle_takeover_active"] = False
            return False

        active_threshold = float(self.config.get("ego_obstacle_safety_risk_threshold", 0.25))
        clear_risk = float(self.config.get("ego_obstacle_takeover_clear_risk", 0.10))
        trigger_distance = float(self.config.get("ego_obstacle_takeover_distance", 18.0))
        clear_steps_required = int(self.config.get("ego_obstacle_takeover_clear_steps", 8))
        max_steps = int(self.config.get("ego_obstacle_takeover_max_steps", 80))
        distance_active = current_distance < 0.0 or current_distance <= trigger_distance
        should_enter = distance_active and (
            float(current_risk) >= active_threshold
            or escape_option is not None
            or bool(state.get("hazardous_lane_change", False))
            or bool(state.get("lane_change_cancelled_for_obstacle", False))
        )

        if should_enter:
            state["obstacle_takeover_active"] = True
            state["obstacle_takeover_timer"] = 0
            state["obstacle_takeover_clear_steps"] = 0
            state["obstacle_takeover_released"] = False
        elif bool(state.get("obstacle_takeover_active", False)):
            state["obstacle_takeover_timer"] = int(state.get("obstacle_takeover_timer", 0)) + 1
            if float(current_risk) <= clear_risk and state.get("lane_change_intent") is None:
                state["obstacle_takeover_clear_steps"] = int(state.get("obstacle_takeover_clear_steps", 0)) + 1
            else:
                state["obstacle_takeover_clear_steps"] = 0
            if (
                int(state.get("obstacle_takeover_clear_steps", 0)) >= clear_steps_required
                or int(state.get("obstacle_takeover_timer", 0)) >= max_steps
            ):
                state["obstacle_takeover_active"] = False
                state["obstacle_takeover_released"] = True
                state["obstacle_takeover_target_lane_id"] = None
                state["obstacle_takeover_lane_delta"] = 0
                state["obstacle_takeover_timer"] = 0
                state["obstacle_takeover_clear_steps"] = 0

        return bool(state.get("obstacle_takeover_active", False))

    def _cancel_obstacle_hazard_lane_change(
        self,
        actor,
        waypoint,
        state: Dict[str, Any],
        current_risk: float,
        current_distance: float,
    ) -> bool:
        if self._actor_roles.get(actor.id) != "ego" or waypoint is None:
            return False
        target_lane_id = state.get("target_lane_id")
        if target_lane_id is None or int(target_lane_id) == int(getattr(waypoint, "lane_id", 0)):
            return False
        cancel_threshold = float(self.config.get("ego_cancel_lane_change_obstacle_risk", 0.35))
        cancel_distance = float(self.config.get("ego_cancel_lane_change_obstacle_distance", 20.0))
        current_distance_ok = current_distance < 0.0 or current_distance <= cancel_distance
        target_waypoint = None
        for lane_delta in (-1, 1):
            candidate = self._adjacent_driving_waypoint(waypoint, lane_delta)
            if candidate is not None and int(candidate.lane_id) == int(target_lane_id):
                target_waypoint = candidate
                break
        if target_waypoint is None:
            if float(current_risk) < cancel_threshold or not current_distance_ok:
                return False
            self._mark_obstacle_lane_change_cancelled(
                state,
                waypoint,
                float(current_risk),
                current_risk_active=True,
            )
            return True

        target_query = self._lane_obstacle_query_for_actor(actor, target_waypoint)
        target_risk = float(target_query.get("risk", 0.0))
        state["target_lane_obstacle_risk"] = float(max(float(state.get("target_lane_obstacle_risk", 0.0)), target_risk))
        target_distance = float(target_query.get("distance", -1.0))
        within_distance = target_distance < 0.0 or target_distance <= cancel_distance
        improvement_margin = float(self.config.get("ego_obstacle_escape_risk_margin", 0.15))
        target_not_safer = target_risk >= max(0.0, float(current_risk) - improvement_margin)
        target_hazard = target_risk >= cancel_threshold
        current_hazard_without_escape = (
            float(current_risk) >= cancel_threshold
            and current_distance_ok
            and target_not_safer
        )
        if not ((target_hazard and within_distance) or current_hazard_without_escape):
            return False

        self._mark_obstacle_lane_change_cancelled(
            state,
            waypoint,
            max(float(target_risk), float(current_risk) if current_hazard_without_escape else 0.0),
            current_risk_active=float(current_risk) >= float(self.config.get("ego_obstacle_safety_risk_threshold", 0.25)),
        )
        return True

    def _mark_obstacle_lane_change_cancelled(
        self,
        state: Dict[str, Any],
        waypoint,
        risk: float,
        current_risk_active: bool,
    ) -> None:
        state["lane_change_intent"] = None
        state["target_lane_id"] = waypoint.lane_id
        state["lane_change_timer"] = 0
        state["lane_change_failed"] = True
        state["lane_change_success"] = False
        state["lane_change_cancelled_for_obstacle"] = True
        state["lane_obstacle_veto"] = True
        state["hazardous_lane_change"] = True
        state["hazardous_lane_change_risk"] = float(risk)
        state["obstacle_avoidance_blocked_reason"] = "cancelled_for_obstacle"
        if current_risk_active:
            state["lane_change_cooldown"] = 0
        else:
            state["lane_change_cooldown"] = int(self.config.get("ego_obstacle_cancel_cooldown", 8))

    def _can_override_for_obstacle_avoidance(self, state: Dict[str, Any], distance: float) -> bool:
        if state.get("lane_change_intent") is not None:
            return False
        cooldown = int(state.get("lane_change_cooldown", 0))
        if cooldown <= 0:
            return True
        emergency_distance = float(self.config.get("ego_obstacle_emergency_distance", 8.0))
        return 0.0 <= float(distance) <= emergency_distance

    def _target_lane_obstacle_safe(self, current_risk: float, target_risk: float) -> bool:
        veto_threshold = float(self.config.get("ego_lane_obstacle_veto_risk", 0.45))
        margin = float(self.config.get("ego_lane_obstacle_risk_margin", 0.10))
        current_risk = float(current_risk)
        target_risk = float(target_risk)
        if target_risk >= veto_threshold:
            return False
        return target_risk <= current_risk + margin

    def _lane_obstacle_risk_for_actor(self, actor, waypoint):
        query = self._lane_obstacle_query_for_actor(actor, waypoint)
        return float(query["distance"]), float(query["lateral"]), float(query["risk"])

    def _lane_obstacle_query_for_actor(self, actor, waypoint):
        if waypoint is None:
            return {"distance": -1.0, "lateral": 0.0, "risk": 0.0}
        transform = actor.get_transform()
        route_s, _, _ = self._project_to_route(
            np.array([transform.location.x, transform.location.y], dtype=float),
            math.radians(transform.rotation.yaw),
        )
        return self._lane_obstacle_query_at(
            route_s,
            int(getattr(waypoint, "road_id", 0)),
            int(getattr(waypoint, "section_id", 0)),
            int(getattr(waypoint, "lane_id", 0)),
        )

    def _lane_obstacle_risk_for_waypoint(self, actor, waypoint):
        if waypoint is None:
            return 1.0
        transform = actor.get_transform()
        route_s, _, _ = self._project_to_route(
            np.array([transform.location.x, transform.location.y], dtype=float),
            math.radians(transform.rotation.yaw),
        )
        _, _, risk = self._lane_obstacle_risk_at(
            route_s,
            int(getattr(waypoint, "road_id", 0)),
            int(getattr(waypoint, "section_id", 0)),
            int(getattr(waypoint, "lane_id", 0)),
        )
        return float(risk)

    def _lane_obstacle_risk_at(self, route_s: float, road_id: int, section_id: int, raw_lane_id: int):
        query = self._lane_obstacle_query_at(route_s, road_id, section_id, raw_lane_id)
        return float(query["distance"]), float(query["lateral"]), float(query["risk"])

    def _lane_obstacle_query_at(self, route_s: float, road_id: int, section_id: int, raw_lane_id: int):
        if not bool(self.config.get("enable_static_obstacle_awareness", True)):
            return {"distance": -1.0, "lateral": 0.0, "risk": 0.0}
        if not self._static_obstacles:
            return {"distance": -1.0, "lateral": 0.0, "risk": 0.0}
        lookahead = max(float(self.config.get("ego_obstacle_lookahead_distance", 20.0)), 1e-6)
        lateral_tolerance = float(self.config.get("ego_obstacle_lateral_tolerance", 1.5))
        lateral_power = max(float(self.config.get("ego_obstacle_lateral_risk_power", 2.0)), 0.1)
        best = None
        best_risk = 0.0
        for obstacle in self._static_obstacles:
            if (
                int(obstacle["road_id"]) != int(road_id)
                or int(obstacle["section_id"]) != int(section_id)
                or int(obstacle["raw_lane_id"]) != int(raw_lane_id)
            ):
                continue
            distance = float(obstacle["route_s"] - route_s)
            if distance < 0.0 or distance > lookahead:
                continue
            lateral = float(obstacle.get("lane_lateral", obstacle["lateral"]))
            if abs(lateral) > lateral_tolerance:
                continue
            distance_risk = float((1.0 - distance / lookahead) ** 2)
            lateral_overlap = max(0.0, 1.0 - abs(lateral) / max(lateral_tolerance, 1e-6))
            risk = float(distance_risk * (lateral_overlap ** lateral_power))
            if risk > best_risk:
                best = obstacle
                best_risk = risk
        if best is None:
            return {"distance": -1.0, "lateral": 0.0, "risk": 0.0}
        lateral = float(best.get("lane_lateral", best["lateral"]))
        return {
            "distance": float(best["route_s"] - route_s),
            "lateral": lateral,
            "lane_lateral": lateral,
            "risk": float(best_risk),
            "label": str(best.get("label", "")),
            "x": float(best.get("x", 0.0)),
            "y": float(best.get("y", 0.0)),
            "road_id": int(best.get("road_id", 0)),
            "section_id": int(best.get("section_id", 0)),
            "raw_lane_id": int(best.get("raw_lane_id", 0)),
        }

    def _cache_static_obstacles(self):
        self._static_obstacles = []
        self._static_obstacles_ready = True
        if (
            not bool(self.config.get("enable_static_obstacle_awareness", True))
            or self.world is None
            or self.carla is None
            or self._route_points_xy is None
        ):
            return

        world_map = self.world.get_map()
        raw_labels = self.config.get("static_obstacle_labels", ("Poles", "TrafficLight", "TrafficSigns"))
        if isinstance(raw_labels, str):
            labels = [item.strip() for item in raw_labels.split(",") if item.strip()]
        else:
            labels = [str(item) for item in raw_labels]

        seen = set()
        obstacles = []
        for label_name in labels:
            label = getattr(self.carla.CityObjectLabel, label_name, None)
            if label is None:
                continue
            try:
                environment_objects = self.world.get_environment_objects(label)
            except RuntimeError:
                continue
            for environment_object in environment_objects:
                location = self._environment_object_location(environment_object)
                if location is None:
                    continue
                key = (round(float(location.x), 1), round(float(location.y), 1), label_name)
                if key in seen:
                    continue
                seen.add(key)
                waypoint = self._projected_driving_waypoint(location)
                if waypoint is None:
                    continue
                route_s, lateral, _ = self._project_to_route(
                    np.array([location.x, location.y], dtype=float),
                    math.radians(waypoint.transform.rotation.yaw),
                )
                lane_lateral = self._waypoint_lateral_offset(location, waypoint)
                route_end = self._route_total_length + float(self.config.get("ego_obstacle_lookahead_distance", 20.0))
                if route_s < -5.0 or route_s > route_end:
                    continue
                obstacles.append({
                    "route_s": float(route_s),
                    "lateral": float(lateral),
                    "lane_lateral": float(lane_lateral),
                    "road_id": int(getattr(waypoint, "road_id", 0)),
                    "section_id": int(getattr(waypoint, "section_id", 0)),
                    "raw_lane_id": int(getattr(waypoint, "lane_id", 0)),
                    "label": label_name,
                    "x": float(location.x),
                    "y": float(location.y),
                })
        self._static_obstacles = obstacles

    @staticmethod
    def _waypoint_lateral_offset(location, waypoint) -> float:
        waypoint_location = waypoint.transform.location
        yaw = math.radians(waypoint.transform.rotation.yaw)
        dx = float(location.x - waypoint_location.x)
        dy = float(location.y - waypoint_location.y)
        return float(-math.sin(yaw) * dx + math.cos(yaw) * dy)

    def _environment_object_location(self, environment_object):
        transform = getattr(environment_object, "transform", None)
        bounding_box = getattr(environment_object, "bounding_box", None)
        if transform is not None and bounding_box is not None:
            try:
                return transform.transform(bounding_box.location)
            except (AttributeError, RuntimeError):
                pass
        if transform is not None:
            return getattr(transform, "location", None)
        if bounding_box is not None:
            return getattr(bounding_box, "location", None)
        return None

    def _projected_driving_waypoint(self, location):
        try:
            return self.world.get_map().get_waypoint(
                location,
                project_to_road=True,
                lane_type=self.carla.LaneType.Driving,
            )
        except (AttributeError, RuntimeError):
            try:
                return self.world.get_map().get_waypoint(location, project_to_road=True)
            except RuntimeError:
                return None

    def _adjacent_driving_waypoint(self, waypoint, lane_delta: int):
        getter = "get_right_lane" if lane_delta > 0 else "get_left_lane"
        candidate = getattr(waypoint, getter)()
        if candidate is None or not self._is_driving_lane(candidate):
            return None
        return candidate

    def _waypoint_route_aligned(self, waypoint) -> bool:
        if waypoint is None or self._route_points_xy is None or self._route_segment_headings is None:
            return True
        max_error = math.radians(float(self.config.get("ego_obstacle_escape_max_route_heading_error_deg", 25.0)))
        max_lateral = float(self.config.get("ego_obstacle_escape_max_route_lateral", 8.0))
        lookahead = float(self.config.get("ego_obstacle_escape_route_check_lookahead", 8.0))
        candidates = waypoint.next(max(lookahead, 0.5))
        check_waypoint = candidates[0] if candidates else waypoint
        location = check_waypoint.transform.location
        candidate_heading = math.radians(check_waypoint.transform.rotation.yaw)
        _, lateral_offset, heading_error = self._project_to_route(
            np.array([location.x, location.y], dtype=float),
            candidate_heading,
        )
        return abs(float(heading_error)) <= max_error and abs(float(lateral_offset)) <= max_lateral

    def _waypoint_route_heading_error_deg(self, waypoint, lookahead: float) -> float:
        if waypoint is None or self._route_points_xy is None:
            return 0.0
        future_waypoint = self._future_waypoint(waypoint, lookahead)
        location = future_waypoint.transform.location
        heading = math.radians(future_waypoint.transform.rotation.yaw)
        _, _, heading_error = self._project_to_route(
            np.array([location.x, location.y], dtype=float),
            heading,
        )
        return abs(math.degrees(float(heading_error)))

    def _waypoint_route_lateral(self, waypoint) -> float:
        if waypoint is None or self._route_points_xy is None:
            return 0.0
        lookahead = float(self.config.get("ego_obstacle_escape_route_check_lookahead", 8.0))
        future_waypoint = self._future_waypoint(waypoint, lookahead)
        location = future_waypoint.transform.location
        heading = math.radians(future_waypoint.transform.rotation.yaw)
        _, lateral_offset, _ = self._project_to_route(
            np.array([location.x, location.y], dtype=float),
            heading,
        )
        return float(lateral_offset)

    def _actor_route_lateral(self, actor) -> float:
        transform = actor.get_transform()
        _, lateral_offset, _ = self._project_to_route(
            np.array([transform.location.x, transform.location.y], dtype=float),
            math.radians(transform.rotation.yaw),
        )
        return float(lateral_offset)

    @staticmethod
    def _future_waypoint(waypoint, distance: float):
        candidates = waypoint.next(max(float(distance), 0.5))
        return candidates[0] if candidates else waypoint

    def _target_lane_waypoint(self, actor, current_waypoint, target_lane_id):
        if current_waypoint is None or target_lane_id is None:
            return current_waypoint
        if current_waypoint.lane_id == target_lane_id:
            candidates = current_waypoint.next(float(self.config.get("lookahead_distance", 8.0)))
            return candidates[0] if candidates else current_waypoint
        for getter in ("get_left_lane", "get_right_lane"):
            candidate = getattr(current_waypoint, getter)()
            if candidate is not None and self._is_driving_lane(candidate) and candidate.lane_id == target_lane_id:
                next_candidates = candidate.next(float(self.config.get("lookahead_distance", 8.0)))
                return next_candidates[0] if next_candidates else candidate
        return current_waypoint

    def _obstacle_takeover_waypoint(self, actor, current_waypoint, state: Dict[str, Any]):
        if (
            self._actor_roles.get(actor.id) != "ego"
            or current_waypoint is None
            or not bool(state.get("obstacle_takeover_active", False))
        ):
            return None
        lane_delta = int(state.get("obstacle_takeover_lane_delta", 0))
        target_lane_id = state.get("obstacle_takeover_target_lane_id")
        target_waypoint = None
        if lane_delta != 0:
            target_waypoint = self._adjacent_driving_waypoint(current_waypoint, lane_delta)
            if target_waypoint is not None and not self._waypoint_route_aligned(target_waypoint):
                target_waypoint = None
        if target_waypoint is None and target_lane_id is not None:
            for candidate_delta in (-1, 1):
                candidate = self._adjacent_driving_waypoint(current_waypoint, candidate_delta)
                if (
                    candidate is not None
                    and int(candidate.lane_id) == int(target_lane_id)
                    and self._waypoint_route_aligned(candidate)
                ):
                    target_waypoint = candidate
                    state["obstacle_takeover_lane_delta"] = candidate_delta
                    break
        if target_waypoint is None:
            return None
        next_candidates = target_waypoint.next(float(self.config.get("ego_obstacle_takeover_lookahead", 10.0)))
        return next_candidates[0] if next_candidates else target_waypoint

    def _lateral_control(self, actor, target_waypoint):
        transform = actor.get_transform()
        location = transform.location
        target = target_waypoint.transform.location
        yaw = math.radians(transform.rotation.yaw)
        dx = target.x - location.x
        dy = target.y - location.y
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        heading_error = math.radians(target_waypoint.transform.rotation.yaw - transform.rotation.yaw)
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        steer = float(self.config.get("lateral_kp", 0.18)) * local_y + float(self.config.get("heading_kp", 0.8)) * heading_error
        return float(np.clip(steer, -float(self.config.get("max_steer", 0.65)), float(self.config.get("max_steer", 0.65))))

    def _longitudinal_control(self, current_speed: float, target_speed: float):
        error = float(target_speed - current_speed)
        if error >= 0.0:
            throttle = float(np.clip(0.20 + float(self.config.get("speed_kp", 0.08)) * error, 0.0, 1.0))
            return throttle, 0.0
        brake = float(np.clip(float(self.config.get("brake_kp", 0.12)) * abs(error), 0.0, 1.0))
        return 0.0, brake

    @staticmethod
    def _actor_longitudinal_gap(actor, other) -> float:
        transform = actor.get_transform()
        origin = transform.location
        target = other.get_location()
        yaw = math.radians(transform.rotation.yaw)
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        return float((target.x - origin.x) * forward_x + (target.y - origin.y) * forward_y)

    def _actor_route_gap(self, actor, other) -> float:
        actor_transform = actor.get_transform()
        other_transform = other.get_transform()
        actor_s, _, _ = self._project_to_route(
            np.array([actor_transform.location.x, actor_transform.location.y], dtype=float),
            math.radians(actor_transform.rotation.yaw),
        )
        other_s, _, _ = self._project_to_route(
            np.array([other_transform.location.x, other_transform.location.y], dtype=float),
            math.radians(other_transform.rotation.yaw),
        )
        if self._route_points_xy is not None:
            return float(other_s - actor_s)
        return self._actor_longitudinal_gap(actor, other)

    def _lane_change_safe(self, actor, target_waypoint, lane_delta: int):
        role = self._actor_roles.get(actor.id, "background")
        front_gap_min = float(self.config.get("adv_min_lanechange_gap_front", 12.0))
        rear_gap_min = float(self.config.get("adv_min_lanechange_gap_rear", 9.0))
        if role == "ego":
            front_gap_min = float(self.config.get("ego_min_lanechange_gap_front", 15.0))
            rear_gap_min = float(self.config.get("ego_min_lanechange_gap_rear", 10.0))
        actor_loc = actor.get_location()
        current_wp = self.world.get_map().get_waypoint(actor_loc, project_to_road=True)
        target_lane_id = target_waypoint.lane_id
        front_gap = float("inf")
        rear_gap = float("inf")
        current_front_gap = float("inf")
        for other in self._actors:
            if (
                other.id == actor.id
                or not self._is_alive(other)
            ):
                continue
            other_loc = other.get_location()
            wp = self.world.get_map().get_waypoint(other_loc, project_to_road=True)
            if wp is None:
                continue
            gap = self._actor_route_gap(actor, other)
            same_target_segment = (
                wp.road_id == target_waypoint.road_id
                and wp.section_id == target_waypoint.section_id
                and wp.lane_id == target_lane_id
            )
            if same_target_segment:
                if gap >= 0.0:
                    front_gap = min(front_gap, gap)
                else:
                    rear_gap = min(rear_gap, abs(gap))
            same_current_segment = (
                current_wp is not None
                and wp.road_id == current_wp.road_id
                and wp.section_id == current_wp.section_id
                and wp.lane_id == current_wp.lane_id
            )
            if same_current_segment and gap >= 0.0:
                current_front_gap = min(current_front_gap, gap)

        safe_gap = front_gap >= front_gap_min and rear_gap >= rear_gap_min
        if role != "ego":
            return safe_gap
        front_blocked = current_front_gap <= float(self.config.get("ego_lanechange_trigger_gap", 30.0))
        target_better = front_gap >= current_front_gap + float(self.config.get("ego_lanechange_min_gap_gain", 8.0))
        return safe_gap and (front_blocked or target_better)

    def _sync_runtime_state(self):
        advs = []
        hdvs = []
        backgrounds = []
        ego = None
        for actor in self._actors:
            if not self._is_alive(actor):
                continue
            state = self._vehicle_state_from_actor(actor)
            role = self._actor_roles.get(actor.id, "background")
            if role == "ego":
                ego = state
            elif role == "adv_cav":
                advs.append(state)
            elif role == "hdv":
                hdvs.append(state)
            else:
                backgrounds.append(state)
        self.state.ego_vehicle = ego
        self.state.adv_cav_vehicles = advs
        self.state.hdv_vehicles = hdvs
        self.state.background_vehicles = backgrounds

    def _vehicle_state_from_actor(self, actor) -> VehicleState:
        transform = actor.get_transform()
        location = transform.location
        velocity = actor.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        waypoint = self.world.get_map().get_waypoint(location, project_to_road=True)
        raw_lane_id = int(getattr(waypoint, "lane_id", 0)) if waypoint is not None else 0
        lane_id = int(abs(raw_lane_id))
        road_id = int(getattr(waypoint, "road_id", 0)) if waypoint is not None else 0
        section_id = int(getattr(waypoint, "section_id", 0)) if waypoint is not None else 0
        heading = math.radians(transform.rotation.yaw)
        route_s, lateral_offset, heading_error = self._project_to_route(
            np.array([location.x, location.y], dtype=float), heading
        )
        obstacle_distance, obstacle_lateral, obstacle_risk = self._lane_obstacle_risk_at(
            route_s, road_id, section_id, raw_lane_id
        )
        role = self._actor_roles.get(actor.id, "background")
        index = self._actor_indices.get(actor.id, 0)
        control_state = self._control_states.get(actor.id, {})
        return VehicleState(
            role=role,
            index=index,
            position=np.array([location.x, location.y], dtype=np.float32),
            speed=speed,
            lane_id=max(0, min(lane_id, 3)),
            heading=heading,
            route_s=route_s,
            lateral_offset=lateral_offset,
            heading_error=heading_error,
            road_id=road_id,
            section_id=section_id,
            raw_lane_id=raw_lane_id,
            obstacle_distance=obstacle_distance,
            obstacle_lateral=obstacle_lateral,
            obstacle_risk=obstacle_risk,
            target_lane_obstacle_risk=float(control_state.get("target_lane_obstacle_risk", 0.0)),
            lane_obstacle_veto=bool(control_state.get("lane_obstacle_veto", False)),
            lane_change_cancelled_for_obstacle=bool(control_state.get("lane_change_cancelled_for_obstacle", False)),
            hazardous_lane_change=bool(control_state.get("hazardous_lane_change", False)),
            hazardous_lane_change_risk=float(control_state.get("hazardous_lane_change_risk", 0.0)),
            obstacle_takeover_active=bool(control_state.get("obstacle_takeover_active", False)),
            obstacle_takeover_released=bool(control_state.get("obstacle_takeover_released", False)),
            crashed=actor.id in self._collision_actor_ids,
            on_road=waypoint is not None,
            active=True,
            physical_active=True,
        )

    def _update_runtime_metrics(self):
        vehicles = self.state.vehicles
        active_vehicles = [vehicle for vehicle in vehicles if vehicle.active]
        ego = self.state.ego_vehicle
        self.state.crash = bool(ego is not None and ego.crashed)
        crashed_count = sum(1 for vehicle in vehicles if vehicle.crashed)
        self.state.crash_rate = crashed_count / max(len(vehicles), 1)
        self.state.average_speed = (
            float(np.mean([vehicle.speed for vehicle in active_vehicles])) if active_vehicles else 0.0
        )
        self.state.retired_non_ego_count = 0
        self.state.policy_inactive_non_ego_count = sum(
            int(vehicle.role != "ego" and not vehicle.active and vehicle.physical_active)
            for vehicle in vehicles
        )
        self.state.colliding_adv_indices = tuple(sorted(
            vehicle.index for vehicle in self.state.adv_cav_vehicles if vehicle.crashed
        ))
        if ego is not None:
            self.state.route_completion = self._route_completion(ego)
            self.state.route_completion_full = self._route_completion_full(ego)
            self.state.success = self.state.route_completion >= 1.0 and not ego.crashed
        else:
            self.state.route_completion = 0.0
            self.state.route_completion_full = 0.0
            self.state.success = False

    def _route_completion(self, ego: VehicleState) -> float:
        return float(np.clip(ego.route_s / max(self._route_completion_distance(), 1e-6), 0.0, 1.0))

    def _route_completion_full(self, ego: VehicleState) -> float:
        return float(np.clip(ego.route_s / max(self.state.route_length, 1e-6), 0.0, 1.0))

    def _route_completion_distance(self) -> float:
        mode = str(self.config.get("completion_horizon_mode", "full_route")).lower()
        if mode in {"time_budget", "short_horizon", "episode_budget"}:
            override = self.config.get("completion_horizon_distance", None)
            if override is not None:
                distance = float(override)
            else:
                max_steps = int(self.config.get("max_episode_steps", 200))
                dt = float(getattr(self.state, "dt", self.config.get("fixed_delta_seconds", 0.05)))
                target_speed = float(self.config.get(
                    "completion_target_speed",
                    self.config.get("target_speed_cruise", 18.0),
                ))
                ratio = float(self.config.get("completion_time_budget_ratio", 1.0))
                distance = max_steps * dt * target_speed * ratio
            min_distance = float(self.config.get("completion_min_distance", 50.0))
            distance = max(distance, min_distance)
            distance = min(float(self.state.route_length), float(distance))
        else:
            distance = float(self.state.route_length)
        self.state.route_completion_distance = float(distance)
        return float(distance)

    def _compute_route_length(self, scenario_config: Dict[str, Any]) -> float:
        return float(scenario_config.get("route_length", self.config.get("route_length", 200.0)))

    def _set_route_reference(self, ego_actor, route_length: float):
        self._route_start = self._copy_location(ego_actor.get_location())
        start_waypoint = self.world.get_map().get_waypoint(
            ego_actor.get_location(), project_to_road=True, lane_type=self.carla.LaneType.Driving
        )
        points = []
        route_waypoint = start_waypoint
        backtrack = max(float(self.config.get("route_backtrack_distance", 100.0)), 0.0)
        if start_waypoint is not None and backtrack > 0.0:
            predecessors = list(start_waypoint.previous(backtrack))
            if predecessors:
                route_waypoint = self._select_route_successor(start_waypoint, predecessors)
        if route_waypoint is not None:
            points = self._build_waypoint_route(route_waypoint, route_length + backtrack)
        if len(points) < 2:
            transform = ego_actor.get_transform()
            yaw = math.radians(transform.rotation.yaw)
            points = [
                np.array([self._route_start.x, self._route_start.y], dtype=float),
                np.array([
                    self._route_start.x + route_length * math.cos(yaw),
                    self._route_start.y + route_length * math.sin(yaw),
                ], dtype=float),
            ]

        self._route_points_xy = np.asarray(points, dtype=float)
        segments = np.diff(self._route_points_xy, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        valid = lengths > 1e-6
        if not np.all(valid):
            keep = np.concatenate([[True], valid])
            self._route_points_xy = self._route_points_xy[keep]
            segments = np.diff(self._route_points_xy, axis=0)
            lengths = np.linalg.norm(segments, axis=1)
        self._route_cumulative_s = np.concatenate([[0.0], np.cumsum(lengths)])
        self._route_segment_headings = np.arctan2(segments[:, 1], segments[:, 0])
        self._route_origin_s = 0.0
        origin_s, _, _ = self._project_to_route(
            np.array([self._route_start.x, self._route_start.y], dtype=float),
            math.radians(ego_actor.get_transform().rotation.yaw),
        )
        self._route_origin_s = float(origin_s)
        self._route_total_length = float(route_length)
        self._route_start_xy = self._route_points_xy[0].copy()
        self._route_vector_xy = self._route_points_xy[-1] - self._route_points_xy[0]
        end = self._route_points_xy[-1]
        self._route_end = self.carla.Location(x=float(end[0]), y=float(end[1]), z=self._route_start.z)
        self._cache_static_obstacles()

    def _build_waypoint_route(self, start_waypoint, route_length: float):
        sample_distance = max(float(self.config.get("route_sample_distance", 2.0)), 0.5)
        points = [np.array([start_waypoint.transform.location.x, start_waypoint.transform.location.y], dtype=float)]
        waypoint = start_waypoint
        travelled = 0.0
        max_points = int(max(route_length / sample_distance, 1.0)) + 20
        for _ in range(max_points):
            candidates = list(waypoint.next(sample_distance))
            if not candidates:
                break
            waypoint = self._select_route_successor(waypoint, candidates)
            point = np.array([waypoint.transform.location.x, waypoint.transform.location.y], dtype=float)
            segment_length = float(np.linalg.norm(point - points[-1]))
            if segment_length <= 1e-4:
                continue
            points.append(point)
            travelled += segment_length
            if travelled >= route_length:
                break
        return points

    @staticmethod
    def _select_route_successor(current_waypoint, candidates):
        current_yaw = math.radians(current_waypoint.transform.rotation.yaw)

        def heading_change(candidate):
            candidate_yaw = math.radians(candidate.transform.rotation.yaw)
            return abs(math.atan2(math.sin(candidate_yaw - current_yaw), math.cos(candidate_yaw - current_yaw)))

        return min(candidates, key=heading_change)

    def _project_to_route(self, position, heading: float):
        if self._route_points_xy is None or len(self._route_points_xy) < 2:
            return 0.0, 0.0, 0.0
        point = np.asarray(position, dtype=float)[:2]
        starts = self._route_points_xy[:-1]
        segments = self._route_points_xy[1:] - starts
        lengths_sq = np.sum(segments * segments, axis=1)
        relative = point - starts
        t = np.clip(np.sum(relative * segments, axis=1) / np.maximum(lengths_sq, 1e-8), 0.0, 1.0)
        projections = starts + t[:, None] * segments
        offsets = point - projections
        distances_sq = np.sum(offsets * offsets, axis=1)
        index = int(np.argmin(distances_sq))
        segment_length = math.sqrt(max(float(lengths_sq[index]), 1e-8))
        tangent = segments[index] / segment_length
        lateral = float(tangent[0] * offsets[index, 1] - tangent[1] * offsets[index, 0])
        route_s = float(self._route_cumulative_s[index] + t[index] * segment_length - self._route_origin_s)
        route_heading = float(self._route_segment_headings[index])
        heading_error = float(math.atan2(math.sin(heading - route_heading), math.cos(heading - route_heading)))
        return route_s, lateral, heading_error

    def _copy_location(self, location):
        return self.carla.Location(x=location.x, y=location.y, z=location.z)

    def _tick_runtime(self, label: str = "tick"):
        retries = int(self.config.get("tick_retries", 3))
        last_error = None
        for attempt in range(retries + 1):
            try:
                return self.world.tick()
            except RuntimeError as exc:
                last_error = exc
                if attempt >= retries:
                    break
                try:
                    self.client.set_timeout(float(self.config.get("timeout", 60.0)))
                except RuntimeError:
                    pass
        raise RuntimeError(f"CARLA world.tick timeout during {label} after {retries + 1} attempts: {last_error}") from last_error

    def _cleanup_runtime(self):
        for sensor in self._sensors:
            try:
                if self._is_alive(sensor):
                    sensor.stop()
            except RuntimeError:
                pass
        self._destroy_actors(list(self._sensors) + list(self._actors))
        if self.world is not None and (self._sensors or self._actors):
            try:
                self._tick_runtime("cleanup")
            except RuntimeError:
                pass
        self._sensors = []
        self._actors = []
        self._autopilot_actor_ids = set()
        self._actor_roles = {}
        self._actor_indices = {}
        self._collision_actor_ids = set()
        self._reported_collision_actor_ids = set()
        self._collision_first_step = {}
        self._static_obstacles = []
        self._static_obstacles_ready = False
        self._collision_last_event_step = {}
        self._collision_partner_ids = {}
        self._pending_collision_pairs = set()
        self._active_collision_pairs = set()
        self._mock_active_collision_pairs = set()
        self._control_states = {}
        self._route_start_xy = None
        self._route_vector_xy = None
        self._route_points_xy = None
        self._route_cumulative_s = None
        self._route_segment_headings = None
        self._route_total_length = 0.0

    def _destroy_actors(self, actors):
        mode = str(self.config.get("cleanup_destroy_mode", "sequential")).lower()
        if mode == "batch":
            self._destroy_actors_batch(actors)
            return
        for actor in actors:
            try:
                if self._is_alive(actor):
                    actor.destroy()
            except RuntimeError:
                pass

    def _destroy_actors_batch(self, actors):
        live_actor_ids = []
        for actor in actors:
            try:
                if self._is_alive(actor):
                    live_actor_ids.append(actor.id)
            except RuntimeError:
                pass
        if not live_actor_ids:
            return
        if self.client is not None and self.carla is not None:
            try:
                commands = [self.carla.command.DestroyActor(actor_id) for actor_id in live_actor_ids]
                self.client.apply_batch_sync(commands, True)
                return
            except RuntimeError:
                pass
        for actor in actors:
            try:
                if self._is_alive(actor):
                    actor.destroy()
            except RuntimeError:
                pass

    def _purge_existing_runtime_actors(self):
        if self.world is None:
            return
        try:
            actors = self.world.get_actors()
        except RuntimeError:
            return
        stale = []
        for actor in actors:
            try:
                type_id = str(getattr(actor, "type_id", ""))
                if type_id.startswith("vehicle.") or type_id == "sensor.other.collision":
                    stale.append(actor)
            except RuntimeError:
                pass
        self._destroy_actors(stale)
        if stale:
            try:
                self._tick_runtime("purge_existing_actors")
            except RuntimeError:
                pass
        self._route_origin_s = 0.0

    @staticmethod
    def _is_alive(actor) -> bool:
        return bool(actor is not None and getattr(actor, "is_alive", False))

    @staticmethod
    def _blueprint_int(blueprint, name: str, default: int) -> int:
        if not blueprint.has_attribute(name):
            return default
        attribute = blueprint.get_attribute(name)
        try:
            return int(attribute.as_int())
        except AttributeError:
            return int(str(attribute))

    def _reset_mock(self, scenario_config: Dict[str, Any], seed: Optional[int]):
        self._collision_actor_ids = set()
        self._reported_collision_actor_ids = set()
        self._mock_active_collision_pairs = set()
        rng = np.random.RandomState(seed if seed is not None else scenario_config.get("seed", 0))
        num_cav = int(scenario_config.get("num_cav", 1))
        num_hdv = int(scenario_config.get("num_hdv", 3))
        route_length = float(scenario_config.get("route_length", self.config.get("route_length", 200.0)))
        policy_frequency = float(scenario_config.get("policy_frequency", 5.0))
        dt = 1.0 / max(policy_frequency, 1e-6)

        advs = [
            VehicleState(
                role="adv_cav",
                index=i,
                position=np.array([35.0 + i * 18.0, float(i % 2) * 4.0], dtype=np.float32),
                speed=18.0 + rng.uniform(-1.0, 1.0),
                lane_id=i % 2,
                raw_lane_id=i % 2,
                route_s=35.0 + i * 18.0,
                lateral_offset=0.0,
            )
            for i in range(num_cav)
        ]
        ego = VehicleState(
            role="ego",
            index=0,
            position=np.array([0.0, 0.0], dtype=np.float32),
            speed=20.0,
            lane_id=0,
            raw_lane_id=0,
            route_s=0.0,
            lateral_offset=0.0,
        )
        hdvs = [
            VehicleState(
                role="hdv",
                index=i,
                position=np.array([70.0 + i * 22.0, float((i + 1) % 2) * 4.0], dtype=np.float32),
                speed=17.0 + rng.uniform(-1.0, 1.0),
                lane_id=(i + 1) % 2,
                raw_lane_id=(i + 1) % 2,
                route_s=70.0 + i * 22.0,
                lateral_offset=0.0,
            )
            for i in range(num_hdv)
        ]

        self.state = ScenarioState(
            name=scenario_config.get("route_name", "merge_smoke"),
            ego_vehicle=ego,
            adv_cav_vehicles=advs,
            hdv_vehicles=hdvs,
            route_length=route_length,
            dt=dt,
        )
        self._update_metrics()
        return self.state

    def _step_mock(self, control_intents: Sequence[Dict[str, Any]]):
        self.state.newly_collided_adv_indices = ()
        self.state.recovered_adv_indices = ()
        self.state.newly_collided_non_ego_count = 0
        vehicles = self.state.vehicles
        for vehicle, intent in zip(vehicles, control_intents):
            self._apply_control(vehicle, intent)
        if len(control_intents) < len(vehicles):
            for vehicle in vehicles[len(control_intents):]:
                self._apply_control(vehicle, {"throttle": 0.3, "brake": 0.0, "lane_change": 0})
        self.state.steps += 1
        self._detect_collisions()
        self._update_metrics()
        return self.state

    def _apply_control(self, vehicle: VehicleState, intent: Dict[str, Any]):
        if not vehicle.active:
            return
        if vehicle.crashed and vehicle.role != "ego":
            intent = self._mock_collision_recovery_intent(vehicle)
        throttle = float(intent.get("throttle", 0.0))
        brake = float(intent.get("brake", 0.0))
        lane_change = int(intent.get("lane_change", 0))
        if vehicle.role in {"ego", "adv_cav"}:
            throttle, brake = self._apply_mock_front_safety(vehicle, throttle, brake)
        accel = 2.5 * throttle - 4.0 * brake
        vehicle.speed = float(np.clip(vehicle.speed + accel * self.state.dt, 0.0, 35.0))
        if lane_change:
            vehicle.set_lane(int(np.clip(vehicle.lane_id + lane_change, 0, 2)))
        progress = vehicle.speed * self.state.dt
        vehicle.position[0] += progress
        vehicle.route_s += progress
        vehicle.lateral_offset = 0.0
        vehicle.heading_error = 0.0
        vehicle.on_road = 0 <= vehicle.lane_id <= 2

    def _mock_collision_recovery_intent(self, vehicle: VehicleState):
        preferred = -1 if vehicle.index % 2 == 0 else 1
        target_lane = int(np.clip(vehicle.lane_id + preferred, 0, 2))
        lane_change = preferred if target_lane != vehicle.lane_id else 0
        return {
            "throttle": float(self.config.get("collision_recovery_throttle", 0.35)) if lane_change else 0.0,
            "brake": 0.0 if lane_change else 0.25,
            "lane_change": lane_change,
        }

    def _apply_mock_front_safety(self, vehicle: VehicleState, throttle: float, brake: float):
        front_vehicles = [
            other for other in self.state.vehicles
            if (
                other is not vehicle
                and other.physical_active
                and other.lane_id == vehicle.lane_id
                and longitudinal_gap(vehicle, other) >= 0.0
            )
        ]
        if not front_vehicles:
            return throttle, brake
        front = min(front_vehicles, key=lambda other: longitudinal_gap(vehicle, other))
        front_gap = longitudinal_gap(vehicle, front)
        prefix = "ego" if vehicle.role == "ego" else "adv"
        default_emergency_gap = 8.0 if vehicle.role == "ego" else 5.0
        default_light_gap = 12.0 if vehicle.role == "ego" else 8.0
        default_light_brake = 0.20 if vehicle.role == "ego" else 0.12
        default_emergency_brake = 0.45 if vehicle.role == "ego" else 0.30
        emergency_gap = float(self.config.get(f"{prefix}_front_emergency_gap", default_emergency_gap))
        light_brake_gap = float(self.config.get(f"{prefix}_front_light_brake_gap", default_light_gap))
        if front_gap <= emergency_gap:
            return 0.0, max(
                brake, float(self.config.get(f"{prefix}_front_emergency_brake", default_emergency_brake))
            )
        if front_gap <= light_brake_gap and vehicle.speed > front.speed:
            return 0.0, max(
                brake, float(self.config.get(f"{prefix}_front_light_brake", default_light_brake))
            )
        return throttle, brake

    def _detect_collisions(self):
        vehicles = self.state.vehicles
        current_pairs = set()
        newly_adv = set()
        newly_non_ego_ids = set()
        for idx, vehicle in enumerate(vehicles):
            if not vehicle.physical_active:
                continue
            for other in vehicles[idx + 1:]:
                if not other.physical_active:
                    continue
                same_lane = vehicle.lane_id == other.lane_id
                close_longitudinal = abs(vehicle.x - other.x) < 3.0
                if not (same_lane and close_longitudinal):
                    continue
                left = (vehicle.role, int(vehicle.index))
                right = (other.role, int(other.index))
                pair = tuple(sorted((left, right)))
                current_pairs.add(pair)
                is_new_pair = pair not in self._mock_active_collision_pairs
                for collided in (vehicle, other):
                    collided.crashed = True
                    if collided.role != "ego":
                        collided.active = True
                        if is_new_pair:
                            newly_non_ego_ids.add((collided.role, int(collided.index)))
                        if is_new_pair and collided.role == "adv_cav":
                            newly_adv.add(int(collided.index))
                if is_new_pair and (vehicle.role == "adv_cav" or other.role == "adv_cav"):
                    self.state.adv_collision_event_count += 1
                    roles = {vehicle.role, other.role}
                    if vehicle.role == other.role == "adv_cav":
                        self.state.adv_adv_collision_event_count += 1
                    elif "ego" in roles:
                        self.state.adv_ego_collision_event_count += 1
                    else:
                        self.state.adv_other_collision_event_count += 1

        recovered = []
        current_vehicle_keys = {key for pair in current_pairs for key in pair}
        previous_vehicle_keys = {
            key for pair in self._mock_active_collision_pairs for key in pair
        }
        for vehicle in vehicles:
            key = (vehicle.role, int(vehicle.index))
            if (
                vehicle.role != "ego"
                and key in previous_vehicle_keys
                and key not in current_vehicle_keys
            ):
                vehicle.crashed = False
                vehicle.active = True
                if vehicle.role == "adv_cav":
                    recovered.append(int(vehicle.index))
        self._mock_active_collision_pairs = current_pairs
        self.state.newly_collided_adv_indices = tuple(sorted(newly_adv))
        self.state.colliding_adv_indices = tuple(sorted(
            vehicle.index for vehicle in self.state.adv_cav_vehicles if vehicle.crashed
        ))
        self.state.recovered_adv_indices = tuple(sorted(set(recovered)))
        self.state.adv_recovery_count += len(set(recovered))
        self.state.newly_collided_non_ego_count = int(len(newly_non_ego_ids))

    def _update_metrics(self):
        vehicles = self.state.vehicles
        active_vehicles = [vehicle for vehicle in vehicles if vehicle.active]
        ego = self.state.ego_vehicle
        self.state.crash = bool(ego is not None and ego.crashed)
        self.state.crash_rate = (
            sum(1 for vehicle in vehicles if vehicle.crashed) / max(len(vehicles), 1)
        )
        self.state.average_speed = (
            float(np.mean([vehicle.speed for vehicle in active_vehicles])) if active_vehicles else 0.0
        )
        self.state.retired_non_ego_count = 0
        self.state.policy_inactive_non_ego_count = sum(
            int(vehicle.role != "ego" and not vehicle.active and vehicle.physical_active)
            for vehicle in vehicles
        )
        if ego is not None:
            self.state.route_completion = self._route_completion(ego)
            self.state.route_completion_full = self._route_completion_full(ego)
            self.state.success = self.state.route_completion >= 1.0 and not ego.crashed
        else:
            self.state.route_completion = 0.0
            self.state.route_completion_full = 0.0
            self.state.success = False
