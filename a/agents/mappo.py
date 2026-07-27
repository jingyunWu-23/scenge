"""MAPPO implementation for adversarial CAV training."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import torch as th
    from torch import nn
    from torch.distributions import Categorical
    from torch.optim import Adam, RMSprop
except ImportError as exc:
    raise RuntimeError(
        "MAPPO training requires PyTorch in the Python interpreter running this "
        f"command ({sys.executable}). Activate the training environment or install "
        "a CUDA-compatible torch build before running carla_evolution/training/train.py."
    ) from exc

def _resolve_marl_dir() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "MARL1",
        here.parents[1] / "MARL1",
        Path.cwd().parent / "MARL1",
        Path.cwd() / "MARL1",
    ]
    for candidate in candidates:
        if (candidate / "single_agent" / "Model_common.py").exists():
            return candidate
    return candidates[0]


MARL_DIR = _resolve_marl_dir()
if str(MARL_DIR) not in sys.path:
    sys.path.insert(0, str(MARL_DIR))

from single_agent.Model_common import ActorNetwork  # noqa: E402


class CentralizedCritic(nn.Module):
    """Shared value network using local state plus pooled global agent context."""

    def __init__(self, state_dim: int, hidden_size: int):
        super().__init__()
        input_dim = int(state_dim) * 3
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, centralized_state):
        return self.network(centralized_state)


class MAPPOAgent:
    """Shared discrete actor with a centralized critic and PPO/GAE updates."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        memory_capacity: int = 10000,
        batch_size: int = 120,
        roll_out_n_steps: int = 1000,
        reward_gamma: float = 0.98,
        reward_scale: float = 100.0,
        actor_hidden_size: int = 128,
        critic_hidden_size: int = 128,
        actor_lr: float = 5e-4,
        critic_lr: float = 5e-4,
        entropy_reg: float = 0.01,
        value_coef: float = 0.5,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 10,
        max_grad_norm: float = 0.2,
        target_update_steps: int = 3,
        target_tau: float = 1.0,
        clip_param: float = 0.2,
        optimizer_type: str = "adam",
        reward_type: str = "global_R",
        episodes_before_train: int = 3,
        normalize_advantages: bool = True,
        use_cuda: bool = True,
        torch_seed: int = 669,
    ):
        assert reward_type in {"regionalR", "global_R", "agents_rewards"}
        th.manual_seed(int(torch_seed))
        np.random.seed(int(torch_seed))
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.reward_type = reward_type
        self.reward_gamma = float(reward_gamma)
        self.reward_scale = float(reward_scale)
        self.batch_size = int(batch_size)
        self.roll_out_n_steps = int(roll_out_n_steps)
        self.entropy_reg = float(entropy_reg)
        self.value_coef = float(value_coef)
        self.gae_lambda = float(gae_lambda)
        self.ppo_epochs = int(ppo_epochs)
        self.max_grad_norm = float(max_grad_norm)
        self.clip_param = float(clip_param)
        self.episodes_before_train = int(episodes_before_train)
        self.normalize_advantages = bool(normalize_advantages)
        self.memory_capacity = int(memory_capacity)
        self.use_cuda = bool(use_cuda and th.cuda.is_available())
        self.device = th.device("cuda" if self.use_cuda else "cpu")
        self.n_episodes = 0
        self.n_steps = 0

        # Retain these attributes for CLI/checkpoint compatibility. Standard PPO
        # uses rollout log probabilities instead of target policy networks.
        self.target_update_steps = int(target_update_steps)
        self.target_tau = float(target_tau)

        self.actor = ActorNetwork(
            self.state_dim,
            int(actor_hidden_size),
            self.action_dim,
            nn.functional.log_softmax,
        ).to(self.device)
        self.critic = CentralizedCritic(self.state_dim, int(critic_hidden_size)).to(self.device)

        optimizer_cls = RMSprop if optimizer_type == "rmsprop" else Adam
        self.actor_optimizer = optimizer_cls(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optimizer = optimizer_cls(self.critic.parameters(), lr=float(critic_lr))
        self._rollouts = []

    @classmethod
    def from_config(cls, config: Dict, state_dim: int, action_dim: int):
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            memory_capacity=config.get("memory_capacity", 10000),
            batch_size=config.get("batch_size", 120),
            roll_out_n_steps=config.get("roll_out_n_steps", 1000),
            reward_gamma=config.get("reward_gamma", 0.98),
            reward_scale=config.get("reward_scale", 100.0),
            actor_hidden_size=config.get("actor_hidden_size", 128),
            critic_hidden_size=config.get("critic_hidden_size", 128),
            actor_lr=config.get("actor_lr", 5e-4),
            critic_lr=config.get("critic_lr", 5e-4),
            entropy_reg=config.get("entropy_reg", 0.01),
            value_coef=config.get("value_coef", 0.5),
            gae_lambda=config.get("gae_lambda", 0.95),
            ppo_epochs=config.get("ppo_epochs", 10),
            max_grad_norm=config.get("max_grad_norm", 0.2),
            target_update_steps=config.get("target_update_steps", 3),
            target_tau=config.get("target_tau", 1.0),
            clip_param=config.get("clip_param", 0.2),
            optimizer_type=config.get("optimizer_type", "adam"),
            reward_type=config.get("reward_type", "global_R"),
            episodes_before_train=config.get("episodes_before_train", 3),
            normalize_advantages=config.get("normalize_advantages", True),
            use_cuda=config.get("use_cuda", True),
            torch_seed=config.get("torch_seed", 669),
        )

    def sample_actions(self, state, n_agents: int, deterministic: bool = False) -> Tuple[List[int], np.ndarray, np.ndarray]:
        states = self._state_tensor(state, n_agents)
        with th.no_grad():
            log_probs_all = self.actor(states)
            distribution = Categorical(logits=log_probs_all)
            actions = th.argmax(log_probs_all, dim=-1) if deterministic else distribution.sample()
            log_probs = distribution.log_prob(actions)
            values = self.critic(self._centralized_inputs(states.unsqueeze(0))).squeeze(-1).squeeze(0)
        return (
            actions.cpu().numpy().astype(np.int64).tolist(),
            log_probs.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),
        )

    def exploration_action(self, state, n_agents: int) -> List[int]:
        actions, _, _ = self.sample_actions(state, n_agents, deterministic=False)
        return actions

    def action(self, state, n_agents: int) -> List[int]:
        actions, _, _ = self.sample_actions(state, n_agents, deterministic=True)
        return actions

    def store_episode(
        self,
        states: Sequence,
        actions: Sequence,
        rewards: Sequence,
        log_probs: Sequence = None,
        values: Sequence = None,
        dones: Sequence = None,
        next_states: Sequence = None,
        active_masks: Sequence = None,
    ):
        if not states:
            return
        states_array = np.asarray(states, dtype=np.float32)
        actions_array = self._action_indices(actions)
        rewards_array = np.asarray(rewards, dtype=np.float32)
        if rewards_array.ndim != 2:
            raise ValueError(f"MAPPO rewards must have shape [T, N], got {rewards_array.shape}.")
        steps, n_agents = rewards_array.shape
        if states_array.shape != (steps, n_agents, self.state_dim):
            raise ValueError(
                f"MAPPO states must have shape {(steps, n_agents, self.state_dim)}, got {states_array.shape}."
            )
        if actions_array.shape != (steps, n_agents):
            raise ValueError(f"MAPPO actions must have shape {(steps, n_agents)}, got {actions_array.shape}.")

        if log_probs is None or values is None:
            sampled_log_probs = []
            sampled_values = []
            for state_step, action_step in zip(states_array, actions_array):
                state_tensor = th.as_tensor(state_step, dtype=th.float32, device=self.device)
                action_tensor = th.as_tensor(action_step, dtype=th.long, device=self.device)
                with th.no_grad():
                    distribution = Categorical(logits=self.actor(state_tensor))
                    sampled_log_probs.append(distribution.log_prob(action_tensor).cpu().numpy())
                    sampled_values.append(
                        self.critic(self._centralized_inputs(state_tensor.unsqueeze(0)))
                        .squeeze(-1).squeeze(0).cpu().numpy()
                    )
            log_probs_array = np.asarray(sampled_log_probs, dtype=np.float32)
            values_array = np.asarray(sampled_values, dtype=np.float32)
        else:
            log_probs_array = np.asarray(log_probs, dtype=np.float32)
            values_array = np.asarray(values, dtype=np.float32)

        dones_array = np.asarray(dones if dones is not None else np.zeros_like(rewards_array), dtype=np.float32)
        if dones_array.ndim == 1:
            dones_array = np.repeat(dones_array[:, None], n_agents, axis=1)
        next_states_array = np.asarray(next_states if next_states is not None else states_array, dtype=np.float32)
        if next_states_array.shape != states_array.shape:
            raise ValueError(f"MAPPO next_states must match states shape, got {next_states_array.shape}.")
        active_masks_array = np.asarray(
            active_masks if active_masks is not None else np.ones_like(rewards_array), dtype=np.float32
        )
        if active_masks_array.shape != rewards_array.shape:
            raise ValueError(f"MAPPO active_masks must match rewards shape, got {active_masks_array.shape}.")

        if self.reward_scale > 0:
            rewards_array = rewards_array / self.reward_scale

        with th.no_grad():
            final_next_states = th.as_tensor(next_states_array[-1], dtype=th.float32, device=self.device)
            next_values = (
                self.critic(self._centralized_inputs(final_next_states.unsqueeze(0)))
                .squeeze(-1).squeeze(0).cpu().numpy().astype(np.float32)
            )
        next_values = next_values * (1.0 - dones_array[-1])
        advantages, returns = self._compute_gae(rewards_array, values_array, dones_array, next_values)

        self._rollouts.append({
            "states": states_array,
            "actions": actions_array,
            "old_log_probs": log_probs_array,
            "old_values": values_array,
            "advantages": advantages,
            "returns": returns,
            "active_masks": active_masks_array,
        })
        self.n_episodes += 1
        self._trim_rollouts()

    def train(self):
        if self.n_episodes <= self.episodes_before_train or not self._rollouts:
            return {}

        batch = self._flatten_rollouts()
        self._rollouts = []
        sample_count = batch["actions"].shape[0]
        if sample_count == 0:
            return {}

        advantages = batch["advantages"]
        if self.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        actor_losses = []
        critic_losses = []
        entropy_values = []
        clip_fractions = []
        value_errors = []
        indices = np.arange(sample_count)

        for _ in range(max(self.ppo_epochs, 1)):
            np.random.shuffle(indices)
            for start in range(0, sample_count, max(self.batch_size, 1)):
                batch_indices = th.as_tensor(
                    indices[start:start + max(self.batch_size, 1)], dtype=th.long, device=self.device
                )
                local_states = batch["local_states"][batch_indices]
                centralized_states = batch["centralized_states"][batch_indices]
                actions = batch["actions"][batch_indices]
                old_log_probs = batch["old_log_probs"][batch_indices]
                mini_advantages = advantages[batch_indices]
                returns = batch["returns"][batch_indices]
                old_values = batch["old_values"][batch_indices]

                distribution = Categorical(logits=self.actor(local_states))
                new_log_probs = distribution.log_prob(actions)
                entropy = distribution.entropy().mean()
                ratio = th.exp(new_log_probs - old_log_probs)
                unclipped = ratio * mini_advantages
                clipped = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mini_advantages
                actor_loss = -th.min(unclipped, clipped).mean() - self.entropy_reg * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                predicted_values = self.critic(centralized_states).squeeze(-1)
                clipped_values = old_values + th.clamp(
                    predicted_values - old_values, -self.clip_param, self.clip_param
                )
                value_loss = th.maximum(
                    (predicted_values - returns).pow(2),
                    (clipped_values - returns).pow(2),
                ).mean()
                critic_loss = self.value_coef * value_loss

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropy_values.append(float(entropy.item()))
                clip_fractions.append(float((th.abs(ratio - 1.0) > self.clip_param).float().mean().item()))
                value_errors.append(float(th.mean((predicted_values.detach() - returns).pow(2)).item()))

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "value_error": float(np.mean(value_errors)) if value_errors else 0.0,
            "samples": int(sample_count),
            "ppo_epochs": int(self.ppo_epochs),
        }

    def save(self, model_dir: str, global_step: int):
        os.makedirs(model_dir, exist_ok=True)
        actor_path = os.path.join(model_dir, f"actor_{global_step}.pt")
        critic_path = os.path.join(model_dir, f"critic_{global_step}.pt")
        th.save({
            "global_step": int(global_step),
            "model_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }, actor_path)
        th.save({
            "global_step": int(global_step),
            "model_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.critic_optimizer.state_dict(),
            "centralized_input_dim": self.state_dim * 3,
        }, critic_path)

    def load(self, model_dir: str, global_step: int, train_mode: bool = False):
        actor_path = os.path.join(model_dir, f"actor_{global_step}.pt")
        critic_path = os.path.join(model_dir, f"critic_{global_step}.pt")
        actor_checkpoint = th.load(actor_path, map_location=self.device)
        actor_state = actor_checkpoint.get("model_state_dict", actor_checkpoint)
        self.actor.load_state_dict(actor_state)
        if train_mode and isinstance(actor_checkpoint, dict) and "optimizer_state_dict" in actor_checkpoint:
            self.actor_optimizer.load_state_dict(actor_checkpoint["optimizer_state_dict"])

        critic_checkpoint = th.load(critic_path, map_location=self.device)
        critic_state = critic_checkpoint.get("model_state_dict", critic_checkpoint)
        try:
            self.critic.load_state_dict(critic_state)
        except RuntimeError as exc:
            raise RuntimeError(
                "The critic checkpoint uses the previous local state-action architecture and cannot be loaded "
                "into the centralized critic. Keep the actor checkpoint if needed, but retrain the MAPPO critic."
            ) from exc
        if train_mode and isinstance(critic_checkpoint, dict) and "optimizer_state_dict" in critic_checkpoint:
            self.critic_optimizer.load_state_dict(critic_checkpoint["optimizer_state_dict"])
        self.actor.train(mode=train_mode)
        self.critic.train(mode=train_mode)

    def _state_tensor(self, state, n_agents: int):
        states = np.asarray(state, dtype=np.float32)
        expected = (int(n_agents), self.state_dim)
        if states.shape != expected:
            raise ValueError(f"MAPPO state must have shape {expected}, got {states.shape}.")
        return th.as_tensor(states, dtype=th.float32, device=self.device)

    def _centralized_inputs(self, states):
        """Convert [B, N, D] states to [B, N, 3D] critic inputs."""
        if states.ndim != 3:
            raise ValueError(f"Centralized critic expects [B, N, D], got {tuple(states.shape)}.")
        mean_context = states.mean(dim=1, keepdim=True).expand(-1, states.shape[1], -1)
        max_context = states.max(dim=1, keepdim=True).values.expand(-1, states.shape[1], -1)
        return th.cat([states, mean_context, max_context], dim=-1)

    def _compute_gae(self, rewards, values, dones, final_values):
        steps, n_agents = rewards.shape
        advantages = np.zeros((steps, n_agents), dtype=np.float32)
        gae = np.zeros(n_agents, dtype=np.float32)
        next_values = np.asarray(final_values, dtype=np.float32)
        for step in reversed(range(steps)):
            non_terminal = 1.0 - dones[step]
            delta = rewards[step] + self.reward_gamma * next_values * non_terminal - values[step]
            gae = delta + self.reward_gamma * self.gae_lambda * non_terminal * gae
            advantages[step] = gae
            next_values = values[step]
        return advantages, advantages + values

    def _flatten_rollouts(self):
        states = np.concatenate([rollout["states"] for rollout in self._rollouts], axis=0)
        actions = np.concatenate([rollout["actions"] for rollout in self._rollouts], axis=0)
        old_log_probs = np.concatenate([rollout["old_log_probs"] for rollout in self._rollouts], axis=0)
        old_values = np.concatenate([rollout["old_values"] for rollout in self._rollouts], axis=0)
        advantages = np.concatenate([rollout["advantages"] for rollout in self._rollouts], axis=0)
        returns = np.concatenate([rollout["returns"] for rollout in self._rollouts], axis=0)
        active_masks = np.concatenate([rollout["active_masks"] for rollout in self._rollouts], axis=0)

        states_tensor = th.as_tensor(states, dtype=th.float32, device=self.device)
        centralized = self._centralized_inputs(states_tensor)
        valid = active_masks.reshape(-1) > 0.5
        valid_tensor = th.as_tensor(valid, dtype=th.bool, device=self.device)
        return {
            "local_states": states_tensor.reshape(-1, self.state_dim)[valid_tensor],
            "centralized_states": centralized.reshape(-1, self.state_dim * 3)[valid_tensor],
            "actions": th.as_tensor(actions.reshape(-1)[valid], dtype=th.long, device=self.device),
            "old_log_probs": th.as_tensor(old_log_probs.reshape(-1)[valid], dtype=th.float32, device=self.device),
            "old_values": th.as_tensor(old_values.reshape(-1)[valid], dtype=th.float32, device=self.device),
            "advantages": th.as_tensor(advantages.reshape(-1)[valid], dtype=th.float32, device=self.device),
            "returns": th.as_tensor(returns.reshape(-1)[valid], dtype=th.float32, device=self.device),
        }

    def _trim_rollouts(self):
        total_steps = sum(rollout["states"].shape[0] for rollout in self._rollouts)
        while self._rollouts and total_steps > self.memory_capacity:
            removed = self._rollouts.pop(0)
            total_steps -= removed["states"].shape[0]

    def _action_indices(self, actions):
        array = np.asarray(actions)
        if array.ndim == 3 and array.shape[-1] == self.action_dim:
            return np.argmax(array, axis=-1).astype(np.int64)
        return array.astype(np.int64)
