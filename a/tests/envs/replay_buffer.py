"""Replay buffer for LC straight-road scenario-policy training."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


class LCReplayBuffer:
    """Stores LC init actions and episode rewards.

    The LC policy is on-policy and only needs one decision per episode:
    static scene observation -> initial scenario parameters. Step data is kept
    only to compute the final episode reward.
    """

    def __init__(self, num_scenario: int = 1, mode: str = "train_lc", buffer_capacity: int = 1000):
        self.num_scenario = int(num_scenario)
        self.mode = mode
        self.buffer_capacity = int(buffer_capacity)
        self.buffer_len = 0
        self.reset_buffer()
        self.reset_init_buffer()

    def reset_buffer(self) -> None:
        self.buffer_ego_actions: List[list] = [[] for _ in range(self.num_scenario)]
        self.buffer_scenario_actions: List[list] = [[] for _ in range(self.num_scenario)]
        self.buffer_obs: List[list] = [[] for _ in range(self.num_scenario)]
        self.buffer_next_obs: List[list] = [[] for _ in range(self.num_scenario)]
        self.buffer_rewards: List[list] = [[] for _ in range(self.num_scenario)]
        self.buffer_dones: List[list] = [[] for _ in range(self.num_scenario)]
        self.buffer_additional_dict: List[dict] = [{} for _ in range(self.num_scenario)]

    def reset_init_buffer(self) -> None:
        self.buffer_static_obs: List[np.ndarray] = []
        self.buffer_init_action: List[np.ndarray] = []
        self.buffer_episode_reward: List[float] = []
        self.buffer_init_additional_dict: Dict[str, list] = {}
        self.init_buffer_len = 0

    def store(self, data_list: Iterable[Any], additional_dict: Iterable[Dict[str, Any]]) -> None:
        ego_actions, scenario_actions, obs, next_obs, rewards, dones = data_list
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        self.buffer_len += len(rewards)

        for item_index, info in enumerate(additional_dict):
            sid = int(info.get("scenario_id", 0))
            if sid < 0 or sid >= self.num_scenario:
                raise IndexError(f"scenario_id {sid} outside [0, {self.num_scenario}).")
            self.buffer_ego_actions[sid].append(ego_actions[item_index])
            self.buffer_scenario_actions[sid].append(scenario_actions[item_index])
            self.buffer_obs[sid].append(obs[item_index])
            self.buffer_next_obs[sid].append(next_obs[item_index])
            self.buffer_rewards[sid].append(rewards[item_index])
            self.buffer_dones[sid].append(dones[item_index])

            for key, value in info.items():
                if key == "scenario_id":
                    continue
                self.buffer_additional_dict[sid].setdefault(key, []).append(value)

    def store_init(self, data_list: Iterable[Any], additional_dict: Optional[Dict[str, Any]] = None) -> None:
        static_obs, scenario_init_action = data_list
        static_obs = np.asarray(static_obs, dtype=np.float32)
        scenario_init_action = np.asarray(scenario_init_action, dtype=np.float32)
        self.buffer_static_obs.append(static_obs)
        self.buffer_init_action.append(scenario_init_action)
        self.init_buffer_len += len(scenario_init_action)

        if additional_dict:
            for key, value in additional_dict.items():
                self.buffer_init_additional_dict.setdefault(key, []).append(value)

    def finish_one_episode(self, scenario_ids: Optional[Iterable[int]] = None) -> None:
        scenario_ids = list(range(self.num_scenario)) if scenario_ids is None else [int(i) for i in scenario_ids]
        for sid in scenario_ids:
            dones = np.where(np.asarray(self.buffer_dones[sid], dtype=bool))[0]
            if len(dones) == 0:
                continue
            start = dones[-2] if len(dones) > 1 else -1
            end = dones[-1]
            reward = float(np.sum(self.buffer_rewards[sid][start + 1 : end + 1]))
            self.buffer_episode_reward.append(reward)

    def add_episode_reward(self, episode_reward: float) -> None:
        self.buffer_episode_reward.append(float(episode_reward))

    def sample_init(self, batch_size: int) -> Dict[str, Any]:
        if not self.buffer_init_action:
            raise RuntimeError("LCReplayBuffer has no init actions to sample.")
        if len(self.buffer_episode_reward) < len(self.buffer_init_action):
            raise RuntimeError(
                "LCReplayBuffer cannot sample before each init action has a matching episode reward."
            )

        num_trajectory = len(self.buffer_init_action)
        start_idx = max(0, num_trajectory - self.buffer_capacity)
        sample_index = np.random.randint(0, num_trajectory - start_idx, size=int(batch_size))

        static_obs = np.concatenate(self.buffer_static_obs[start_idx:], axis=0)[sample_index]
        init_action = np.concatenate(self.buffer_init_action[start_idx:], axis=0)[sample_index]
        episode_reward = np.asarray(self.buffer_episode_reward[start_idx:], dtype=np.float32)[sample_index]
        batch: Dict[str, Any] = {
            "static_obs": static_obs,
            "init_action": init_action,
            "episode_reward": episode_reward,
        }

        for key, values in self.buffer_init_additional_dict.items():
            recent_values = values[start_idx:]
            if not recent_values:
                continue
            first = recent_values[0]
            if torch.is_tensor(first):
                batch[key] = torch.cat(recent_values, dim=0)[sample_index]
            else:
                batch[key] = np.asarray(recent_values)[sample_index]
        return batch


# Compatibility alias for copied SafeBench call sites.
RouteReplayBuffer = LCReplayBuffer
