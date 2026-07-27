"""EgoPPO adapter for ego policy training."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Dict


def _resolve_marl_dir() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "MARL1",
        here.parents[1] / "MARL1",
        Path.cwd().parent / "MARL1",
        Path.cwd() / "MARL1",
    ]
    for candidate in candidates:
        if (candidate / "ego" / "config.py").exists() and (candidate / "ego" / "ppo.py").exists():
            return candidate
    return candidates[0]


MARL_DIR = _resolve_marl_dir()
EGO_DIR = MARL_DIR / "ego"
if str(MARL_DIR) not in sys.path:
    sys.path.insert(0, str(MARL_DIR))

# Avoid executing MARL1/ego/__init__.py because it imports the old gym wrapper.
# The CARLA migration only reuses the algorithm modules below.
if "ego" not in sys.modules:
    ego_pkg = types.ModuleType("ego")
    ego_pkg.__path__ = [str(EGO_DIR)]
    sys.modules["ego"] = ego_pkg

try:
    PPOConfig = importlib.import_module("ego.config").PPOConfig
    EgoPPO = importlib.import_module("ego.ppo").EgoPPO
except ImportError as exc:
    raise RuntimeError(
        "EgoPPO import failed. Check that PyTorch is installed in the Python "
        f"interpreter running this command ({sys.executable}) and that MARL1 is "
        f"available at {MARL_DIR}. Original error: {type(exc).__name__}: {exc}"
    ) from exc


class EgoPPOAdapter:
    """Thin wrapper around the original EgoPPO implementation."""

    def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
        config = config or {}
        self.config = PPOConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space_type="discrete",
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            reward_gamma=config.get("reward_gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            ent_coef=config.get("ent_coef", 0.01),
            vf_coef=config.get("vf_coef", 0.5),
            actor_lr=config.get("actor_lr", 3e-4),
            critic_lr=config.get("critic_lr", 3e-4),
            actor_hidden_size=config.get("actor_hidden_size", 128),
            critic_hidden_size=config.get("critic_hidden_size", 128),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            use_cuda=config.get("use_cuda", True),
            seed=config.get("seed", 669),
        )
        self.policy = EgoPPO(self.config)

    @classmethod
    def from_config(cls, config: Dict, state_dim: int, action_dim: int):
        return cls(state_dim=state_dim, action_dim=action_dim, config=config)

    def select_action(self, state, deterministic: bool = False):
        return self.policy.select_action(state, deterministic=deterministic)

    def step(self, state, action, log_prob, value, reward, done, next_state):
        self.policy.step(state, action, log_prob, value, reward, done, next_state)

    def update(self):
        return self.policy.update()

    def end_episode(self):
        return self.policy.end_episode()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.policy.save(path)

    def load(self, path: str):
        self.policy.load(path)
