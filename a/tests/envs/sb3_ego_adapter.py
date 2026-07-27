"""Adapters for evaluating SB3 ego policies in the SafeBench-style tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SB3ContinuousEgoAsDiscretePolicy:
    """Expose a continuous SB3 policy through the old discrete EgoPPO API.

    The migrated CARLA-CAT policy observes a history vector and outputs
    [throttle_brake, steer]. The SafeBench-style test harness expects 25-D
    observations and one of five high-level discrete driving actions. This
    adapter pads/truncates the 25-D test observation to the SB3 observation
    space and maps the continuous action to the nearest high-level intent.
    """

    def __init__(self, checkpoint: str | Path):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("stable-baselines3 is required to evaluate SB3 ego .zip checkpoints") from exc
        self.checkpoint = str(checkpoint)
        self.model = PPO.load(self.checkpoint)
        self.obs_dim = self._obs_dim()

    def select_action(self, state, deterministic: bool = True):
        obs = self._adapt_obs(state)
        continuous_action, _ = self.model.predict(obs, deterministic=deterministic)
        discrete_action = self._continuous_to_discrete(continuous_action)
        return int(discrete_action), None, None

    def _obs_dim(self) -> int:
        shape = getattr(self.model.observation_space, "shape", None)
        if not shape:
            return 25
        return int(np.prod(shape))

    def _adapt_obs(self, state) -> np.ndarray:
        obs = np.asarray(state, dtype=np.float32).reshape(-1)
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        elif len(obs) > self.obs_dim:
            obs = obs[: self.obs_dim]
        return obs.astype(np.float32)

    @staticmethod
    def _continuous_to_discrete(action) -> int:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        throttle_brake = float(arr[0]) if len(arr) > 0 else 0.0
        steer = float(arr[1]) if len(arr) > 1 else 0.0
        if steer < -0.25:
            return 0  # LANE_LEFT
        if steer > 0.25:
            return 2  # LANE_RIGHT
        if throttle_brake > 0.20:
            return 3  # ACCELERATE
        if throttle_brake < -0.20:
            return 4  # DECELERATE
        return 1  # KEEP_LANE


def is_sb3_checkpoint(path: str | Path | None) -> bool:
    return bool(path) and str(path).endswith(".zip")
