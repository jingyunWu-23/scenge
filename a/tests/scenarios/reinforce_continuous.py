"""
Standalone LC (Learning to Collide) policy for straight-road tests.

This keeps SafeBench's core idea: a scenario policy samples initial scene
parameters and is trained with the negative ego episode reward.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


def _as_device(device: Optional[Any] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_routes(routes: np.ndarray) -> np.ndarray:
    routes = np.asarray(routes, dtype=np.float32)
    mean_x = np.mean(routes[:, 0:1])
    max_x = max(float(np.max(np.abs(routes[:, 0:1]))), 1e-6)
    x_1_2 = (routes[:, 0:1] - mean_x) / max_x

    mean_y = np.mean(routes[:, 1:2])
    max_y = max(float(np.max(np.abs(routes[:, 1:2]))), 1e-6)
    y_1_2 = (routes[:, 1:2] - mean_y) / max_y

    return np.concatenate([x_1_2, y_1_2], axis=0)


class IndependantModel(nn.Module):
    def __init__(self, num_waypoint: int = 20):
        super().__init__()
        input_size = num_waypoint * 2 + 1
        hidden_size_1 = 64

        self.a_os = 1
        self.b_os = 1
        self.c_os = 1
        self.d_os = 1

        self.relu = nn.ReLU()
        self.fc_input = nn.Sequential(nn.Linear(input_size, hidden_size_1))
        self.fc_action_a = nn.Sequential(nn.Linear(hidden_size_1, self.a_os * 2))
        self.fc_action_b = nn.Sequential(nn.Linear(1 + hidden_size_1, self.b_os * 2))
        self.fc_action_c = nn.Sequential(nn.Linear(1 + 1 + hidden_size_1, self.c_os * 2))
        self.fc_action_d = nn.Sequential(nn.Linear(1 + 1 + 1 + hidden_size_1, self.d_os * 2))

    @staticmethod
    def sample_action(normal_action: torch.Tensor, action_os: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = normal_action[:, :action_os]
        sigma = F.softplus(normal_action[:, action_os:]) + 1e-6
        eps = torch.randn_like(mu)
        action = mu + sigma * eps
        return action, mu, sigma

    def forward(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del deterministic
        s = self.relu(self.fc_input(x))
        action_a, mu_a, sigma_a = self.sample_action(self.fc_action_a(s), self.a_os)
        action_b, mu_b, sigma_b = self.sample_action(self.fc_action_b(s), self.b_os)
        action_c, mu_c, sigma_c = self.sample_action(self.fc_action_c(s), self.c_os)
        action_d, mu_d, sigma_d = self.sample_action(self.fc_action_d(s), self.d_os)
        action = torch.cat((action_a, action_b, action_c, action_d), dim=1)
        mu = torch.cat((mu_a, mu_b, mu_c, mu_d), dim=1)
        sigma = torch.cat((sigma_a, sigma_b, sigma_c, sigma_d), dim=1)
        return mu, sigma, action


class AutoregressiveModel(nn.Module):
    def __init__(self, num_waypoint: int = 30, standard_action_dim: bool = True):
        super().__init__()
        self.standard_action_dim = bool(standard_action_dim)
        input_size = num_waypoint * 2 + 1
        hidden_size_1 = 32

        self.a_os = 1
        self.b_os = 1
        self.c_os = 1
        self.d_os = 1 if self.standard_action_dim else 0

        self.relu = nn.ReLU()
        self.fc_input = nn.Sequential(nn.Linear(input_size, hidden_size_1))
        self.fc_action_a = nn.Sequential(nn.Linear(hidden_size_1, self.a_os * 2))
        self.fc_action_b = nn.Sequential(nn.Linear(1 + hidden_size_1, self.b_os * 2))
        self.fc_action_c = nn.Sequential(nn.Linear(1 + 1 + hidden_size_1, self.c_os * 2))
        if self.standard_action_dim:
            self.fc_action_d = nn.Sequential(nn.Linear(1 + 1 + 1 + hidden_size_1, self.d_os * 2))

    @staticmethod
    def sample_action(normal_action: torch.Tensor, action_os: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = normal_action[:, :action_os]
        sigma = F.softplus(normal_action[:, action_os:]) + 1e-6
        action = mu + sigma * torch.randn_like(mu)
        return action, mu, sigma

    def forward(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.relu(self.fc_input(x))

        action_a, mu_a, sigma_a = self.sample_action(self.fc_action_a(s), self.a_os)

        state_a = torch.cat((s, mu_a if deterministic else action_a), dim=1)
        action_b, mu_b, sigma_b = self.sample_action(self.fc_action_b(state_a), self.b_os)

        state_ab = torch.cat((s, mu_a if deterministic else action_a, mu_b if deterministic else action_b), dim=1)
        action_c, mu_c, sigma_c = self.sample_action(self.fc_action_c(state_ab), self.c_os)

        actions = [action_a, action_b, action_c]
        mus = [mu_a, mu_b, mu_c]
        sigmas = [sigma_a, sigma_b, sigma_c]

        if self.standard_action_dim:
            state_abc = torch.cat(
                (s, mu_a if deterministic else action_a, mu_b if deterministic else action_b, mu_c if deterministic else action_c),
                dim=1,
            )
            action_d, mu_d, sigma_d = self.sample_action(self.fc_action_d(state_abc), self.d_os)
            actions.append(action_d)
            mus.append(mu_d)
            sigmas.append(sigma_d)

        return torch.cat(mus, dim=1), torch.cat(sigmas, dim=1), torch.cat(actions, dim=1)


class LCReinforcePolicy:
    name = "lc_reinforce"
    type = "init_state"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,
        device: Optional[Any] = None,
    ):
        config = config or {}
        self.logger = logger
        self.device = _as_device(device or config.get("device"))
        self.num_waypoint = int(config.get("num_waypoint", 30))
        self.num_scenario = int(config.get("num_scenario", 1))
        self.batch_size = int(config.get("batch_size", 32))
        self.lr = float(config.get("lr", 1e-3))
        self.entropy_weight = float(config.get("entropy_weight", 1e-4))
        self.model_id = config.get("model_id", 0)
        self.model_path = Path(config.get("model_path", "tests/results/lc"))
        root_dir = config.get("ROOT_DIR")
        if root_dir and not self.model_path.is_absolute():
            self.model_path = Path(root_dir) / self.model_path
        self.standard_action_dim = bool(config.get("standard_action_dim", True))
        self.mode = "train"

        self.model = AutoregressiveModel(self.num_waypoint, self.standard_action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _log(self, message: str) -> None:
        if self.logger is None:
            return
        log_fn = getattr(self.logger, "log", None)
        if callable(log_fn):
            log_fn(message)

    def train(self, replay_buffer: Any) -> Optional[Dict[str, float]]:
        if replay_buffer.init_buffer_len < self.batch_size:
            return None

        batch = replay_buffer.sample_init(self.batch_size)
        episode_reward = torch.as_tensor(batch["episode_reward"], dtype=torch.float32, device=self.device)
        log_prob = batch["log_prob"].to(self.device)
        entropy = batch["entropy"].to(self.device)

        lc_reward = -episode_reward
        loss = (log_prob * lc_reward - entropy * self.entropy_weight).mean(dim=0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        replay_buffer.reset_init_buffer()

        result = {
            "loss": float(loss.detach().cpu().item()),
            "episode_reward_mean": float(episode_reward.detach().cpu().mean().item()),
            "lc_reward_mean": float(lc_reward.detach().cpu().mean().item()),
        }
        self._log(f">> LC training loss: {result['loss']:.4f}")
        return result

    def set_mode(self, mode: str) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(f"Unknown mode {mode}")
        self.mode = mode
        self.model.train(mode == "train")

    def process_init_state(self, state: Iterable[Dict[str, Any]]) -> np.ndarray:
        processed_state_list = []
        for item in state:
            route = np.asarray(item["route"], dtype=np.float32)
            if route.ndim != 2 or route.shape[1] < 2:
                raise ValueError(f"LC init state route must have shape [N, >=2], got {route.shape}.")
            if len(route) < 2:
                raise ValueError("LC init state route must contain at least two waypoints.")

            target_speed = float(item.get("target_speed", 10.0)) / 10.0
            index = np.linspace(1, len(route) - 1, self.num_waypoint).astype(int)
            route_norm = normalize_routes(route[index, :2])[:, 0]
            processed_state = np.concatenate((route_norm, [target_speed]), axis=0).astype(np.float32)
            processed_state_list.append(processed_state)
        return np.stack(processed_state_list, axis=0)

    # Backward-compatible name from SafeBench.
    proceess_init_state = process_init_state

    def get_action(self, state: Any, infos: Any = None, deterministic: bool = False):
        del state, infos, deterministic
        return [None] * self.num_scenario

    def get_init_action(self, state: Iterable[Dict[str, Any]], deterministic: bool = False):
        processed_state = torch.from_numpy(self.process_init_state(state)).to(self.device)
        with torch.set_grad_enabled(self.mode == "train"):
            mu, sigma, action = self.model.forward(processed_state, deterministic)
            action_dist = Normal(mu, sigma)
            log_prob = action_dist.log_prob(action).sum(dim=1)
            entropy = (0.5 * torch.log(2 * torch.pi * sigma.pow(2)) + 0.5).sum(dim=1)

        clipped_action = torch.clamp(action, -1.0, 1.0).detach().cpu().numpy()
        return clipped_action, {"log_prob": log_prob, "entropy": entropy}

    def load_model(self, path: Optional[str] = None) -> bool:
        model_filename = Path(path) if path else self.model_path / f"{self.model_id}.pt"
        if not model_filename.exists():
            self._log(f">> LC model not found: {model_filename}")
            return False
        checkpoint = torch.load(model_filename, map_location=self.device)
        parameters = checkpoint.get("parameters", checkpoint)
        self.model.load_state_dict(parameters)
        self._log(f">> Loaded LC model from {model_filename}")
        return True

    def save_model(self, epoch: Optional[int] = None, path: Optional[str] = None) -> str:
        if path:
            model_filename = Path(path)
        else:
            suffix = f"{self.model_id}.pt" if epoch is None else f"{self.model_id}_epoch{epoch}.pt"
            model_filename = self.model_path / suffix
        os.makedirs(model_filename.parent, exist_ok=True)
        torch.save({"parameters": self.model.state_dict()}, model_filename)
        self._log(f">> Saved LC model to {model_filename}")
        return str(model_filename)


# Compatibility alias for code that still imports SafeBench's class name.
REINFORCE = LCReinforcePolicy
