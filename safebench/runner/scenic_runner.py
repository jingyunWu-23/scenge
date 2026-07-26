"""ScenicRunner: drives the Scenic / ChatScene / ScenGE pipeline.

Worlds are obtained via ``client.get_world()`` after Scenic has loaded the
map. ``train_scenario`` mode is implemented as ``eval(select=True)``; the
former ``train()`` method that referenced undefined ``train_episode`` was
broken and has been removed.
"""

from __future__ import annotations

import csv
import copy
import json
import os

import numpy as np

from safebench.agent.safe_rl.worker.buffer import OnPolicyBuffer
from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.runner.base_runner import BaseRunner
from safebench.scenario.scenario_data_loader import ScenicDataLoader
from safebench.scenario.tools.scenario_utils import scenic_parse
from safebench.util.scenic_utils import ScenicSimulator


class ScenicRunner(BaseRunner):
    """Scenic-driven runner used by the Scenic / ChatScene / ScenGE pipelines."""

    def _prepare_mode(self):
        self.traj_root = self.scenario_config.get("traj_root")
        self.record_traj = bool(self.scenario_config.get("record_traj", False))
        self.replay = bool(self.scenario_config.get("replay", False))
        self._active_replay_tracks = {}
        self._active_replay_step = 0
        if self.mode in ("train_scenario", "eval"):
            self.save_freq = self.agent_config["save_freq"]
            if self.mode == "eval":
                self.logger.log(">> Evaluation Mode, skip config saving", "yellow")
            self.logger.create_eval_dir(load_existing_results=False)
        elif self.mode == "train_ego":
            self.save_freq = self.agent_config["save_freq"]
            self.logger.create_training_dir()
            self.logger.save_config(
                {"agent": self.agent_config, "scenario": self.scenario_config}
            )
        elif self.mode == "eval_ego":
            self.save_freq = self.agent_config["save_freq"]
            self.logger.log(">> Ego Evaluation Mode", "yellow")
            self.logger.create_eval_dir(load_existing_results=False)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

    # ---- world ---------------------------------------------------------

    def _init_world(self):
        self.logger.log(">> Initializing carla world")
        self.world = self.client.get_world()
        self._apply_world_settings(self.world)

    def _init_scenic(self, config):
        self.logger.log(f">> Initializing scenic simulator: {config.scenic_file}")
        self.scenic = ScenicSimulator(config.scenic_file, config.extra_params)

    def run_scenes(self, scenes):
        self.logger.log(">> Begin to run the scene...")
        for scene in scenes:
            if self.scenic.setSimulation(scene):
                self.scenic.update_behavior = self.scenic.runSimulation()
                next(self.scenic.update_behavior)

    # ---- ScenGE trajectory handoff --------------------------------------

    @staticmethod
    def _scenario_stem(log_name, data_id):
        return f"{log_name}_scene-{data_id}"

    @staticmethod
    def _as_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _record_to_step(cls, record, actor_key, step_idx):
        if actor_key == "ego":
            return {
                "t": cls._as_float(record.get("current_game_time", step_idx), step_idx),
                "x": cls._as_float(record.get("ego_x")),
                "y": cls._as_float(record.get("ego_y")),
                "z": cls._as_float(record.get("ego_z")),
                "v": cls._as_float(record.get("ego_velocity")),
                "yaw": cls._as_float(record.get("ego_yaw")),
            }
        actor_record = record.get(actor_key)
        if actor_record is None:
            return None
        return {
            "t": cls._as_float(record.get("current_game_time", step_idx), step_idx),
            "x": cls._as_float(actor_record.get("x")),
            "y": cls._as_float(actor_record.get("y")),
            "z": cls._as_float(actor_record.get("z")),
            "v": cls._as_float(actor_record.get("velocity")),
            "yaw": cls._as_float(actor_record.get("yaw")),
        }

    def _write_traj_json(self, records, stem):
        if not self.record_traj or not self.traj_root or not records:
            return
        env_keys = sorted(k for k in records[0] if k.startswith("env_actor_"))
        actor_sources = [("ego", "ego"), ("adv", "adv_agent_0")]
        actor_sources.extend((key, key) for key in env_keys)
        if "adv_agent_0" not in records[0]:
            self.logger.log(
                f">> Skip trajectory dump for {stem}: missing adv actor",
                color="yellow",
            )
            return
        if not env_keys:
            self.logger.log(
                f">> Skip trajectory dump for {stem}: no env actors to perturb",
                color="yellow",
            )
            return

        traj = {out_key: [] for out_key, _ in actor_sources}
        for step_idx, record in enumerate(records):
            for out_key, source_key in actor_sources:
                step = self._record_to_step(record, source_key, step_idx)
                if step is not None:
                    traj[out_key].append(step)

        traj_dir = os.path.join(self.traj_root, "traj_dir")
        os.makedirs(traj_dir, exist_ok=True)
        out_path = os.path.join(traj_dir, f"{stem}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(traj, fh, indent=2)
        self.logger.log(f">> Saved ScenGE trajectory JSON to {out_path}")

    def _load_replay_tracks(self, stem):
        if not self.replay or not self.traj_root:
            return {}
        replay_dir = os.path.join(self.traj_root, "perturbed_traj", f"perturbed_{stem}")
        if not os.path.isdir(replay_dir):
            self.logger.log(f">> Replay directory not found: {replay_dir}", color="yellow")
            return {}
        tracks = {}
        for name in sorted(os.listdir(replay_dir)):
            if not name.endswith(".csv"):
                continue
            actor_key = os.path.splitext(name)[0]
            with open(os.path.join(replay_dir, name), "r", newline="") as fh:
                tracks[actor_key] = list(csv.DictReader(fh))
        return tracks

    def _activate_replay(self, stem):
        self._active_replay_tracks = self._load_replay_tracks(stem)
        self._active_replay_step = 0

    def _before_env_step(self):
        if not self._active_replay_tracks or self.env is None:
            return None
        if not self.env.env_list:
            return None
        carla_env = self.env.env_list[0]._env
        scenario = carla_env.scenario_manager.background_scenario
        actor_map = {
            f"env_actor_{idx}": actor for idx, actor in enumerate(scenario.env_actors)
        }
        for actor_key, rows in self._active_replay_tracks.items():
            actor = actor_map.get(actor_key)
            if actor is None or self._active_replay_step >= len(rows):
                continue
            row = rows[self._active_replay_step]
            transform = actor.get_transform()
            transform.location.x = self._as_float(row.get("x"), transform.location.x)
            transform.location.y = self._as_float(row.get("y"), transform.location.y)
            transform.location.z = self._as_float(row.get("z"), transform.location.z)
            transform.rotation.yaw = self._as_float(row.get("yaw"), transform.rotation.yaw)
            actor.set_transform(transform)
        self._active_replay_step += 1
        return None

    # ---- eval ----------------------------------------------------------

    def eval(self, data_loader, select=False):
        num_finished_scenario = 0
        data_loader.reset_idx_counter()

        behavior_name = data_loader.behavior
        route_id = data_loader.route_id
        opt_step = data_loader.opt_step
        opt_time = 0

        if route_id is None:
            log_name = f"OPT_{behavior_name}"
        else:
            log_name = f"OPT_{behavior_name}_ROUTE-{route_id}"

        if select:
            self.scene_map[log_name] = {}
            self.scene_map[log_name][f"opt_time_{opt_time}"] = self.scenic.save_params()

        sampled_scenario_configs = []
        infos = []
        while len(data_loader) > 0:
            sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
            num_finished_scenario += num_sampled_scenario
            assert (
                num_sampled_scenario == 1
            ), "scenic can only run one scene at one time"

            data_ids = [config.data_id for config in sampled_scenario_configs]
            scenes = [config.scene for config in sampled_scenario_configs]
            self.run_scenes(scenes)

            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            self.scenario_policy.load_model(sampled_scenario_configs)
            scenario_init_action, _ = self.scenario_policy.get_init_action(
                static_obs, deterministic=True
            )
            obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)
            stem = self._scenario_stem(log_name, data_ids[0])
            self._activate_replay(stem)
            self.env.pre_tick_callback = self._before_env_step

            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            score_list = {s_i: [] for s_i in range(num_sampled_scenario)}
            _, infos = self._run_step_loop(
                obs,
                infos,
                deterministic=True,
                score_list=score_list,
            )

            if self.save_video:
                self.logger.save_video(data_ids=data_ids, log_name=log_name)

            for data_id in data_ids:
                records = self.env.running_results.get(data_id)
                if records is not None:
                    self._write_traj_json(records, self._scenario_stem(log_name, data_id))

            self.logger.log(">> All scenarios are completed. Cleaning up all actors")
            self.env.clean_up()
            self._active_replay_tracks = {}

            self.logger.log(
                f"[{num_finished_scenario}/{data_loader.num_total_scenario}] "
                "Ranking scores for batch scenario:",
                color="yellow",
            )
            for s_i in score_list:
                self.logger.log(
                    "\t Env id " + str(s_i) + ": " + str(np.mean(score_list[s_i])),
                    color="yellow",
                )

            all_running_results = self.logger.add_eval_results(
                records=self.env.running_results
            )
            all_scores = self._route_scores(all_running_results)
            self.logger.add_eval_results(scores=all_scores)
            self.logger.print_eval_results()
            if len(self.env.running_results) % self.save_freq == 0:
                self.logger.save_eval_results(log_name)

            if infos and infos[0]["collision"]:
                self.scenic.record_params()
            if select and (num_finished_scenario % opt_step == 0):
                opt_time += 1
                self.scenic.update_params()
                self.scene_map[log_name][
                    f"opt_time_{opt_time}"
                ] = self.scenic.save_params()
                data_loader.train_scene(opt_time)

        self.logger.save_eval_results(log_name)

        if select:
            from safebench.util.metric_util import get_route_scores

            self.scene_map[log_name]["select_id"] = self.select_adv_scene(
                self.logger.eval_records, get_route_scores, data_loader.select_num
            )
            if sampled_scenario_configs:
                self.dump_scene_map(sampled_scenario_configs[0].scenario_id)

        self.logger.clear()
        self.scenic.destroy()

    # ---- PPO ego training ---------------------------------------------

    def _ppo_env_actions(self, obs, deterministic=False):
        actions, values, logps = self.agent_policy.act(
            obs, deterministic=deterministic, with_logprob=True
        )
        env_actions = actions.copy()
        env_actions[:, 1] = -env_actions[:, 1]
        return actions, env_actions, values, logps

    def _train_ego_episode(self, sampled_scenario_configs, data_ids, run_label=None):
        static_obs = self.env.get_static_obs(sampled_scenario_configs)
        self.scenario_policy.load_model(sampled_scenario_configs)
        scenario_init_action, _ = self.scenario_policy.get_init_action(
            static_obs, deterministic=True
        )
        obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)
        self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

        buffer_size = self.scenario_config["max_episode_step"] + 1
        worker_cfg = self.agent_config["ppo"]["worker_config"]
        replay_buffer = OnPolicyBuffer(
            self.agent_config["ego_state_dim"],
            self.agent_config["ego_action_dim"],
            buffer_size,
            gamma=worker_cfg.get("gamma", self.agent_config["ppo"]["gamma"]),
            lam=worker_cfg.get("lam", 0.97),
        )

        episode_reward = []
        step_count = 0
        while not self.env.all_scenario_done():
            raw_actions, env_actions, values, logps = self._ppo_env_actions(
                obs, deterministic=False
            )
            scenario_actions = self.scenario_policy.get_action(
                obs, infos, deterministic=True
            )
            next_obs, rewards, dones, infos = self.env.step(
                ego_actions=env_actions, scenario_actions=scenario_actions
            )
            replay_buffer.store(
                obs[0],
                raw_actions[0],
                rewards[0],
                values[0],
                logps[0],
                dones[0],
            )
            episode_reward.append(float(rewards[0]))
            step_count += 1
            self._grab_video_frame()
            obs = next_obs

        replay_buffer.finish_path(last_val=0)
        self.agent_policy.policy.learn_on_batch(replay_buffer.get())

        for data_id in data_ids:
            records = self.env.running_results.get(data_id)
            if records is not None:
                result_key = f"{run_label}_scene-{data_id}" if run_label else data_id
                all_running_results = self.logger.add_eval_results(
                    records={result_key: records}
                )
                all_scores = self._route_scores(all_running_results)
                self.logger.add_eval_results(scores=all_scores)

        self.env.clean_up()
        return float(np.sum(episode_reward)), step_count

    def train_ego(self, data_loader, start_epoch=0):
        if self.agent_config["policy_name"] != "ppo":
            raise NotImplementedError("Scenic ego training currently supports PPO only.")

        self.agent_policy.set_mode("train")
        self.scenario_policy.set_mode("eval")
        train_episode = int(self.agent_config["train_episode"])

        for epoch in range(start_epoch + 1, train_episode + 1):
            data_loader.reset_idx_counter()
            epoch_rewards = []
            epoch_steps = 0
            epoch_scenes = 0
            while len(data_loader) > 0:
                sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
                assert (
                    num_sampled_scenario == 1
                ), "scenic can only run one scene at one time"
                data_ids = [config.data_id for config in sampled_scenario_configs]
                scenes = [config.scene for config in sampled_scenario_configs]
                self.run_scenes(scenes)

                episode_reward, step_count = self._train_ego_episode(
                    sampled_scenario_configs, data_ids
                )
                epoch_rewards.append(episode_reward)
                epoch_steps += step_count
                epoch_scenes += num_sampled_scenario

            mean_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
            self.logger.add_training_results("epoch_reward", mean_reward)
            self.logger.add_training_results("epoch_steps", epoch_steps)
            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("Scenes", epoch_scenes)
            self.logger.log_tabular("Steps", epoch_steps)
            self.logger.log_tabular("MeanEpisodeReward", mean_reward)
            self.logger.dump_tabular()

            if epoch % self.save_freq == 0:
                self.agent_policy.save_model(epoch)
                self.logger.save_training_results()

        self.logger.save_training_results()
        self.logger.save_eval_results("train_ego")

    def _config_label(self, config):
        route = "all" if config.route_id is None else config.route_id
        return f"scenario-{config.scenario_id}_{config.behavior}_route-{route}"

    def _setup_scenic_runtime(self, config, last_town):
        self._init_scenic(config)
        if last_town != config.extra_params["town"]:
            self._init_world()
            self._init_renderer(num_panels=2)
            last_town = config.extra_params["town"]
        self.world.scenic = self.scenic
        self.env = VectorWrapper(
            self.env_params,
            self.scenario_config,
            self.world,
            self.birdeye_render,
            self.display,
            self.logger,
        )
        return last_town

    def train_ego_mixed(self, config_list):
        if self.agent_config["policy_name"] != "ppo":
            raise NotImplementedError("Scenic ego training currently supports PPO only.")

        self.agent_policy.set_mode("train")
        self.scenario_policy.set_mode("eval")
        train_episode = int(self.agent_config["train_episode"])
        last_town = None

        self.logger.log(
            f">> Mixed ego training over {len(config_list)} scenic config(s).",
            color="yellow",
        )

        for epoch in range(1, train_episode + 1):
            shuffled_configs = list(config_list)
            np.random.shuffle(shuffled_configs)
            epoch_rewards = []
            epoch_steps = 0
            epoch_scenes = 0

            for config_idx, base_config in enumerate(shuffled_configs):
                config = copy.deepcopy(base_config)
                config.randomize_scenes = True
                config.scene_seed_base = (
                    int(self.scenario_config.get("seed", 0)) * 1000000
                    + epoch * 10000
                    + config_idx * 100
                )
                run_label = self._config_label(config)
                self.logger.log(f">> Training ego on {run_label}")
                try:
                    last_town = self._setup_scenic_runtime(config, last_town)
                    data_loader = ScenicDataLoader(self.scenic, config, self.num_scenario)
                    while len(data_loader) > 0:
                        sampled_scenario_configs, num_sampled_scenario = (
                            data_loader.sampler()
                        )
                        assert (
                            num_sampled_scenario == 1
                        ), "scenic can only run one scene at one time"
                        data_ids = [item.data_id for item in sampled_scenario_configs]
                        scenes = [item.scene for item in sampled_scenario_configs]
                        self.run_scenes(scenes)
                        episode_reward, step_count = self._train_ego_episode(
                            sampled_scenario_configs, data_ids, run_label=run_label
                        )
                        epoch_rewards.append(episode_reward)
                        epoch_steps += step_count
                        epoch_scenes += num_sampled_scenario
                finally:
                    if hasattr(self, "scenic"):
                        self.scenic.destroy()

            mean_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
            self.logger.add_training_results("epoch_reward", mean_reward)
            self.logger.add_training_results("epoch_steps", epoch_steps)
            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("Scenes", epoch_scenes)
            self.logger.log_tabular("Steps", epoch_steps)
            self.logger.log_tabular("MeanEpisodeReward", mean_reward)
            self.logger.dump_tabular()

            if epoch % self.save_freq == 0:
                self.agent_policy.save_model(epoch)
                self.logger.save_training_results()

        self.logger.save_training_results()
        self.logger.save_eval_results("train_ego_mixed")

    def _eval_ego_episode(self, sampled_scenario_configs, data_ids, run_label):
        static_obs = self.env.get_static_obs(sampled_scenario_configs)
        self.scenario_policy.load_model(sampled_scenario_configs)
        scenario_init_action, _ = self.scenario_policy.get_init_action(
            static_obs, deterministic=True
        )
        obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)
        self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

        episode_reward = []
        step_count = 0
        while not self.env.all_scenario_done():
            ego_actions = self.agent_policy.get_action(
                obs, infos, deterministic=True
            )
            scenario_actions = self.scenario_policy.get_action(
                obs, infos, deterministic=True
            )
            obs, rewards, dones, infos = self.env.step(
                ego_actions=ego_actions, scenario_actions=scenario_actions
            )
            episode_reward.append(float(rewards[0]))
            step_count += 1
            self._grab_video_frame()

        for data_id in data_ids:
            records = self.env.running_results.get(data_id)
            if records is not None:
                result_key = f"{run_label}_scene-{data_id}"
                all_running_results = self.logger.add_eval_results(
                    records={result_key: records}
                )
                all_scores = self._route_scores(all_running_results)
                self.logger.add_eval_results(scores=all_scores)

        self.env.clean_up()
        return float(np.sum(episode_reward)), step_count

    def eval_ego_mixed(self, config_list):
        self.agent_policy.load_model()
        self.agent_policy.set_mode("eval")
        self.scenario_policy.set_mode("eval")

        total_episodes = int(self.scenario_config.get("eval_episodes", 10))
        last_town = None
        episode_rewards = []
        episode_steps = 0

        self.logger.log(
            f">> Visual ego eval: {total_episodes} episode(s) over "
            f"{len(config_list)} scenic config(s).",
            color="yellow",
        )

        eval_order = np.random.choice(
            len(config_list), size=total_episodes, replace=True
        )
        for episode, config_idx in enumerate(eval_order):
            config = copy.deepcopy(config_list[config_idx])
            config.sample_num = 1
            config.randomize_scenes = True
            config.scene_seed_base = (
                int(self.scenario_config.get("seed", 0)) * 1000000 + episode * 100
            )
            run_label = f"eval-{episode:03d}_{self._config_label(config)}"
            self.logger.log(f">> Eval ego on {run_label}")
            try:
                last_town = self._setup_scenic_runtime(config, last_town)
                data_loader = ScenicDataLoader(self.scenic, config, self.num_scenario)
                sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
                assert (
                    num_sampled_scenario == 1
                ), "scenic can only run one scene at one time"
                data_ids = [item.data_id for item in sampled_scenario_configs]
                scenes = [item.scene for item in sampled_scenario_configs]
                self.run_scenes(scenes)
                episode_reward, step_count = self._eval_ego_episode(
                    sampled_scenario_configs, data_ids, run_label
                )
                episode_rewards.append(episode_reward)
                episode_steps += step_count
            finally:
                if hasattr(self, "scenic"):
                    self.scenic.destroy()

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        self.logger.log_tabular("Episodes", total_episodes)
        self.logger.log_tabular("Steps", episode_steps)
        self.logger.log_tabular("MeanEpisodeReward", mean_reward)
        self.logger.dump_tabular()
        self.logger.save_eval_results("eval_ego_mixed")

    # ---- adversarial scene selection -----------------------------------

    @staticmethod
    def select_adv_scene(results, score_function, select_num):
        map_id_score_collision = {}
        map_id_score_non_collision = {}
        for i in results.keys():
            score = score_function({i: results[i]})
            if score["collision_rate"] == 1:
                map_id_score_collision[i] = score["final_score"]
            else:
                map_id_score_non_collision[i] = score["final_score"]

        collision_scenes_sorted = sorted(
            map_id_score_collision.items(), key=lambda x: x[1]
        )
        num_collision_selected = min(select_num, len(collision_scenes_sorted))
        selected_scene_id = [
            scene[0] for scene in collision_scenes_sorted[:num_collision_selected]
        ]

        num_non_collision_selected = select_num - num_collision_selected
        if num_non_collision_selected > 0:
            non_collision_scenes_sorted = sorted(
                map_id_score_non_collision.items(), key=lambda x: x[1]
            )
            selected_scene_id.extend(
                scene[0]
                for scene in non_collision_scenes_sorted[:num_non_collision_selected]
            )
        return sorted(selected_scene_id)

    # ---- top-level orchestration ---------------------------------------

    def run(self, test_epoch=None):
        config_list = scenic_parse(self.scenario_config, self.logger)

        if self.mode == "train_ego" and self.scenario_config.get("mixed_training"):
            self.train_ego_mixed(config_list)
            return
        if self.mode == "eval_ego":
            self.eval_ego_mixed(config_list)
            return

        if self.mode == "eval" and test_epoch:
            self.agent_policy.load_model(episode=test_epoch)

        last_town = None
        for config in config_list:
            log_name = (
                f"OPT_{config.behavior}"
                if config.route_id is None
                else f"OPT_{config.behavior}_ROUTE-{config.route_id}"
            )

            if self.mode == "eval":
                if self.logger.check_eval_dir(log_name) == config.select_num:
                    self.logger.log(">> This scenario and route have been done.")
                    continue

            self._init_scenic(config)

            if last_town != config.extra_params["town"]:
                self._init_world()
                self._init_renderer(num_panels=2)
                last_town = config.extra_params["town"]
            self.world.scenic = self.scenic

            self.env = VectorWrapper(
                self.env_params,
                self.scenario_config,
                self.world,
                self.birdeye_render,
                self.display,
                self.logger,
            )

            data_loader = ScenicDataLoader(self.scenic, config, self.num_scenario)

            if self.mode == "train_scenario":
                self.scene_map = self.load_scene_map(config.scenario_id)
                self.agent_policy.set_mode("eval")
                self.scenario_policy.set_mode("eval")
                self.eval(data_loader, select=True)
            elif self.mode == "train_ego":
                self.train_ego(data_loader)
            elif self.mode == "eval":
                self.agent_policy.set_mode("eval")
                self.scenario_policy.set_mode("eval")
                self.eval(data_loader)
            else:
                raise NotImplementedError(f"Unsupported mode: {self.mode}.")

    # ---- scene map persistence -----------------------------------------

    def dump_scene_map(self, scenario_id):
        scenic_dir = os.path.join(
            self.scenario_config["scenic_dir"], f"scenario_{scenario_id}"
        )
        os.makedirs(scenic_dir, exist_ok=True)
        out_path = os.path.join(scenic_dir, f"{os.path.basename(scenic_dir)}.json")
        with open(out_path, "w") as fh:
            json.dump(self.scene_map, fh, indent=4)

    def load_scene_map(self, scenario_id):
        scenic_dir = os.path.join(
            self.scenario_config["scenic_dir"], f"scenario_{scenario_id}"
        )
        out_path = os.path.join(scenic_dir, f"{os.path.basename(scenic_dir)}.json")
        try:
            with open(out_path, "r") as fh:
                return json.loads(fh.read())
        except (OSError, ValueError):
            return {}
