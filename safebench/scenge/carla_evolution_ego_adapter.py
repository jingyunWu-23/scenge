"""Adapters for running SafeBench ego policies inside CARLA Evolution evals."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import types
from typing import Iterable

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_policy_import_path() -> None:
    agent_dir = REPO_ROOT / "safebench" / "agent"
    safe_rl_dir = agent_dir / "safe_rl"
    policy_dir = safe_rl_dir / "policy"
    for name, path in (
        ("safebench.agent", agent_dir),
        ("safebench.agent.safe_rl", safe_rl_dir),
        ("safebench.agent.safe_rl.policy", policy_dir),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module


_ensure_policy_import_path()
from safebench.agent.safe_rl.policy.ppo import PPO


class _AdapterLogger:
    def log(self, msg, color="green"):
        print(msg, flush=True)

    def store(self, **kwargs):
        return None


def _parse_indices(raw: str | None, obs_dim: int) -> list[int]:
    if not raw:
        return list(range(obs_dim))
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(indices) != obs_dim:
        raise ValueError(
            f"Expected {obs_dim} observation indices, got {len(indices)} from {raw!r}"
        )
    return indices


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SafeBenchPPOToDiscreteAdapter:
    """Expose SafeBench PPO as the EgoPPOAdapter interface expected upstream.

    SafeBench PPO emits continuous ``[acceleration, steer]`` actions. CARLA
    Evolution's ego API expects one of five discrete high-level actions:
    lane-left, keep-lane, lane-right, accelerate, decelerate. This adapter maps
    the continuous command into that discrete action set.
    """

    def __init__(self, state_dim: int, action_dim: int, config: dict | None = None):
        config = config or {}
        root = Path(config.get("repo_root") or os.environ.get("SCENGE_ROOT") or Path.cwd())
        agent_cfg = config.get("agent_cfg") or os.environ.get("SAFEBENCH_EGO_AGENT_CFG", "ppo.yaml")
        cfg_path = Path(agent_cfg).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = root / "safebench" / "agent" / "config" / cfg_path
        self.agent_config = _load_yaml(cfg_path)
        self.policy_name = self.agent_config.get("policy_name", "ppo")
        if self.policy_name != "ppo":
            raise ValueError(f"Only SafeBench PPO is supported, got {self.policy_name!r}")
        if not os.environ.get("MODEL_DEVICE"):
            use_cuda = bool(config.get("use_cuda", True)) and torch.cuda.is_available()
            os.environ["MODEL_DEVICE"] = "cuda:0" if use_cuda else "cpu"

        self.obs_dim = int(config.get("safebench_obs_dim") or self.agent_config.get("ego_state_dim", 4))
        self.obs_indices = _parse_indices(
            config.get("obs_indices") or os.environ.get("SAFEBENCH_EGO_OBS_INDICES"),
            self.obs_dim,
        )
        self.steer_threshold = float(
            config.get("steer_threshold")
            or os.environ.get("SAFEBENCH_EGO_STEER_THRESHOLD", 0.1)
        )
        self.acc_threshold = float(
            config.get("acc_threshold")
            or os.environ.get("SAFEBENCH_EGO_ACC_THRESHOLD", 0.5)
        )

        policy_config = dict(self.agent_config[self.policy_name])
        policy_config["ego_state_dim"] = self.obs_dim
        policy_config["ego_action_dim"] = int(self.agent_config.get("ego_action_dim", 2))
        policy_config["ego_action_limit"] = float(self.agent_config.get("ego_action_limit", 3.0))
        self.policy = PPO(policy_config, _AdapterLogger())

    @classmethod
    def from_config(cls, config: dict, state_dim: int, action_dim: int):
        merged = dict(config or {})
        merged.setdefault("repo_root", os.environ.get("SCENGE_ROOT"))
        merged.setdefault("agent_cfg", os.environ.get("SAFEBENCH_EGO_AGENT_CFG", "ppo.yaml"))
        return cls(state_dim=state_dim, action_dim=action_dim, config=merged)

    def load(self, path: str):
        self.policy.load_model(path)

    def select_action(self, state, deterministic: bool = False):
        obs = self._adapt_observation(state)
        action, value, log_prob = self.policy.act(obs, deterministic=deterministic)
        # Match RLAgent.get_action: SafeBench flips steering before CARLA control.
        if len(action) >= 2:
            action = np.asarray(action, dtype=np.float32).copy()
            action[1] = -action[1]
        discrete = self._continuous_to_discrete(action)
        return discrete, log_prob, value

    def _adapt_observation(self, state) -> np.ndarray:
        flat = np.asarray(state, dtype=np.float32).reshape(-1)
        if len(flat) < max(self.obs_indices) + 1:
            padded = np.zeros(max(self.obs_indices) + 1, dtype=np.float32)
            padded[: len(flat)] = flat
            flat = padded
        return flat[self.obs_indices].astype(np.float32, copy=False)

    def _continuous_to_discrete(self, action: Iterable[float]) -> int:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        acc = float(values[0]) if len(values) > 0 else 0.0
        steer = float(values[1]) if len(values) > 1 else 0.0
        if steer <= -self.steer_threshold:
            return 0
        if steer >= self.steer_threshold:
            return 2
        if acc >= self.acc_threshold:
            return 3
        if acc <= -self.acc_threshold:
            return 4
        return 1
