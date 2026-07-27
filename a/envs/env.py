"""Gymnasium-style co-evolution environment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .action_adapter import ActionAdapter
from .metrics import EpisodeMetrics
from .observation import LowDimObservationBuilder
from .reward import RewardManager
from .scenario_manager import ScenarioManager


class ObservationType:
    """Small compatibility object for old wrappers that call observe()."""

    def __init__(self, env: "EvolutionEnv"):
        self.env = env

    def observe(self):
        scenario = self.env.scenario_manager.state
        obs = []
        obs.extend(self.env.observation_builder.build_multi_agent_observation(scenario))
        obs.append(self.env.obs2)
        obs.extend(self.env.obs_hdv_list)
        return obs


class EvolutionEnv:
    """Environment boundary for ego, adversarial CAV, and HDV training."""

    n_a = 5
    n_s = 25

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.action_adapter = ActionAdapter(self.config)
        self.observation_builder = LowDimObservationBuilder(state_dim=self.n_s)
        self.reward_manager = RewardManager(self.config)
        self.metrics = EpisodeMetrics()
        self.scenario_manager = ScenarioManager(self.config)
        self.observation_type = ObservationType(self)
        self.controlled_vehicles = []
        self.ego_vehicle = None
        self.hdv_vehicles = []
        self.obs2 = np.zeros(self.n_s, dtype=np.float32)
        self.obs3 = np.zeros((5, 5), dtype=np.float32)
        self.obs_hdv_list = []
        self.road = None
        self._step_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self._step_count = 0
        self.metrics.reset()
        self.reward_manager.reset()
        scenario = self.scenario_manager.reset(seed=seed, options=options)
        self._sync_from_scenario(scenario)
        obs = self.observation_builder.build_multi_agent_observation(scenario)
        return obs, {
            "scenario": scenario.name,
            "route_completion": float(scenario.route_completion),
            "route_completion_full": float(getattr(scenario, "route_completion_full", scenario.route_completion)),
            "route_completion_distance": float(getattr(scenario, "route_completion_distance", scenario.route_length)),
            "route_length": float(getattr(scenario, "route_length", 0.0)),
            "num_vehicles": len(scenario.vehicles),
        }

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        control_intents = self.action_adapter.to_control_intent_batch(action)
        scenario = self.scenario_manager.step(control_intents)
        self._sync_from_scenario(scenario)
        obs = self.observation_builder.build_multi_agent_observation(scenario)
        reward_info = self.reward_manager.compute(scenario=scenario, action=action)
        terminated = bool(scenario.crash or scenario.success)
        truncated = self._step_count >= self.config.get("max_episode_steps", 200)
        scenario.timeout = bool(truncated and not terminated)
        metrics = self.metrics.update(scenario=scenario, reward=reward_info.reward)
        control_info = self.scenario_manager.control_debug_info()
        info = {**metrics, **reward_info.info, **control_info}
        return obs, float(reward_info.reward), terminated, truncated, info

    def render(self, mode="human", **kwargs):
        if mode == "rgb_array":
            return np.zeros((240, 320, 3), dtype=np.uint8)
        return None

    def close(self):
        self.scenario_manager.close()

    def _sync_from_scenario(self, scenario):
        self.controlled_vehicles = scenario.adv_cav_vehicles
        self.ego_vehicle = scenario.ego_vehicle
        self.hdv_vehicles = scenario.hdv_vehicles
        self.road = scenario.road
        self.obs2 = self.observation_builder.build_ego_observation(scenario)
        self.obs_hdv_list = self.observation_builder.build_hdv_observations(scenario)
        self.obs3 = self.obs_hdv_list[0] if self.obs_hdv_list else np.zeros((5, 5), dtype=np.float32)
