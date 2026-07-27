"""Training entry points for the CARLA evolution package."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import deque
from datetime import datetime

import numpy as np

from carla_evolution.agents.ego_ppo import EgoPPOAdapter
from carla_evolution.agents.mappo import MAPPOAgent
from carla_evolution.envs.action_adapter import DiscreteDrivingAction
from carla_evolution.envs.factory import make_env


ACTION_NAMES = {action.name.lower(): int(action.value) for action in DiscreteDrivingAction}
ACTION_NAMES.update({str(int(action.value)): int(action.value) for action in DiscreteDrivingAction})


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA evolution training")
    parser.add_argument("--mode", choices=["adv", "ego", "joint"], default="adv")
    parser.add_argument("--backend", choices=["mock", "carla"], default="mock")
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-dir", default="模型/carla_evolution/results")
    parser.add_argument("--num-cav", type=int, default=3)
    parser.add_argument("--num-hdv", type=int, default=0)
    parser.add_argument("--num-background", type=int, default=0)
    parser.add_argument("--hdv-model", default=None, help="Frozen SB3 HDV policy .zip to include in joint training.")
    parser.add_argument("--hdv-action", choices=sorted(ACTION_NAMES), default="keep_lane", help="Fallback fixed HDV action.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--ego-action", choices=sorted(ACTION_NAMES), default="accelerate")
    parser.add_argument("--seed", type=int, default=669)
    parser.add_argument("--no-scenario-randomization", action="store_true", default=False)
    parser.add_argument("--scenario-seed-span", type=int, default=1000000)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--adv-episodes", type=int, default=100)
    parser.add_argument("--ego-episodes", type=int, default=50)
    parser.add_argument(
        "--joint-strategy",
        choices=["fixed", "adaptive", "staged-pairs"],
        default="adaptive",
    )
    parser.add_argument(
        "--start-phase",
        choices=["adv", "ego"],
        default="ego",
        help="First phase for adaptive alternating training.",
    )
    parser.add_argument("--round-offset", type=int, default=0)
    parser.add_argument(
        "--stage-cycles",
        default="10,15,25",
        help="Comma-separated staged-pairs cycle counts for initial,middle,final. Use 0,0,N to run only final-stage episodes.",
    )
    parser.add_argument("--resume-adv-model-dir", default=None)
    parser.add_argument("--resume-adv-step", type=int, default=None)
    parser.add_argument("--resume-ego-checkpoint", default=None)
    parser.add_argument("--initial-adv-episodes", type=int, default=0)
    parser.add_argument("--initial-ego-episodes", type=int, default=0)
    parser.add_argument("--adv-min-eps", type=int, default=150)
    parser.add_argument("--adv-max-eps", type=int, default=300)
    parser.add_argument("--ego-min-eps", type=int, default=300)
    parser.add_argument("--ego-max-eps", type=int, default=500)
    parser.add_argument("--adv-crash-threshold", type=float, default=0.15)
    parser.add_argument("--adv-window", type=int, default=50)
    parser.add_argument("--ego-window", type=int, default=100)
    parser.add_argument("--ego-improvement-threshold", type=float, default=0.01)
    parser.add_argument("--ppo-update-interval", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--no-progressive", action="store_true", default=False)
    parser.add_argument("--ego-warmup", action="store_true", default=False)
    parser.add_argument("--ego-warmup-min-episodes", type=int, default=300)
    parser.add_argument("--ego-warmup-max-episodes", type=int, default=3000)
    parser.add_argument("--ego-warmup-target-crash-rate", type=float, default=0.45)
    parser.add_argument("--ego-warmup-target-completion", type=float, default=0.34)
    parser.add_argument("--ego-warmup-window", type=int, default=100)
    parser.add_argument("--warmup-adv-action", choices=sorted(ACTION_NAMES), default="keep_lane")
    parser.add_argument(
        "--ego-fixed-adv-action",
        choices=sorted(ACTION_NAMES),
        default="keep_lane",
        help="Fixed adversarial CAV action used by --mode ego when no resumed MAPPO policy is supplied.",
    )
    parser.add_argument("--ego-batch-size", type=int, default=64)
    parser.add_argument("--ego-n-steps", type=int, default=2048)
    parser.add_argument("--ego-lr", type=float, default=3e-4)

    parser.add_argument("--memory-capacity", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--reward-gamma", type=float, default=0.98)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--actor-hidden-size", type=int, default=128)
    parser.add_argument("--critic-hidden-size", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--entropy-reg", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--mappo-ppo-epochs", type=int, default=10)
    parser.add_argument(
        "--adv-policy-buffer-size",
        type=int,
        default=50,
        help="FIFO buffer size for recent MAPPO adversarial policies used to initialize ego-phase opponents. <=0 disables it.",
    )
    parser.add_argument(
        "--adv-policy-buffer-include-initial",
        action="store_true",
        default=True,
        help="Insert the loaded initial MAPPO policy into the adversarial policy buffer.",
    )
    parser.add_argument(
        "--no-adv-policy-buffer-include-initial",
        action="store_false",
        dest="adv_policy_buffer_include_initial",
        help="Do not insert the loaded initial MAPPO policy into the adversarial policy buffer.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.2)
    parser.add_argument("--target-update-steps", type=int, default=3)
    parser.add_argument("--target-tau", type=float, default=1.0)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--reward-type", choices=["global_R", "regionalR", "agents_rewards"], default="global_R")
    parser.add_argument("--episodes-before-train", type=int, default=3)
    parser.add_argument("--torch-seed", type=int, default=669)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--carla-rpc-timeout", type=float, default=180.0, help="CARLA client RPC timeout used by the training environment.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "adv":
        train_adversarial(args)
    elif args.mode == "ego":
        train_ego(args)
    elif args.mode == "joint":
        train_joint(args)


def episode_seed(args, episode_index: int) -> int:
    if args.no_scenario_randomization:
        return int(args.seed + episode_index)
    span = max(int(args.scenario_seed_span), 1)
    rng = np.random.RandomState(int(args.seed) + int(episode_index) * 9973)
    return int(rng.randint(0, span))


def train_adversarial(args):
    output_dir = make_output_dir(args)
    model_dir = args.model_dir or os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    env_config = {
        "backend": args.backend,
        "max_episode_steps": args.max_steps,
        "num_cav": args.num_cav,
        "num_hdv": args.num_hdv,
        "num_background": args.num_background,
        "timeout": args.carla_rpc_timeout,
        "randomize_scenarios": not args.no_scenario_randomization,
    }
    env = make_env(env_config, config_path=args.config)
    agent = MAPPOAgent.from_config(agent_config(args), state_dim=env.n_s, action_dim=env.n_a)
    ego_action = ACTION_NAMES[args.ego_action]

    log_path = os.path.join(output_dir, "adv_train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields())
        writer.writeheader()

    print("=" * 70)
    print("Adversarial MAPPO pretraining")
    print(f"  backend:        {args.backend}")
    print(f"  num_cav:        {args.num_cav}")
    print(f"  episodes:       {args.episodes}")
    print(f"  max_steps:      {args.max_steps}")
    print(f"  ego_action:     {args.ego_action}")
    print(f"  reward_type:    {args.reward_type}")
    print(f"  randomize:      {not args.no_scenario_randomization}")
    print(f"  output_dir:     {output_dir}")
    print("=" * 70)

    try:
        episode_records = []
        for episode in range(args.episodes):
            metrics = run_adv_episode(env, agent, ego_action, episode_seed(args, episode), args.max_steps, args.reward_type)
            losses = agent.train()
            episode_records.append(metrics)
            row = episode_log_row(episode + 1, metrics, losses)
            append_log(log_path, row)

            if args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0:
                summary = summarize_phase(episode_records[-args.eval_interval:])
                print(
                    f"episode {episode + 1:04d} | "
                    f"reward={summary['avg_reward']:.3f} "
                    f"crash_rate={summary['crash_rate']:.1%} "
                    f"route={summary['road_completion_rate']:.3f} "
                    f"timeout_rate={summary['timeout_rate']:.1%} "
                    f"loss={losses}"
                )
            if args.save_interval > 0 and (episode + 1) % args.save_interval == 0:
                agent.save(model_dir, episode + 1)
        agent.save(model_dir, args.episodes)
    finally:
        env.close()


def train_ego(args):
    """Train only the PPO ego policy with CatSafetyEgoReward."""

    output_dir = make_output_dir(args, prefix="ego")
    model_dir = args.model_dir or os.path.join(output_dir, "models", "ego")
    os.makedirs(model_dir, exist_ok=True)

    env_config = {
        "backend": args.backend,
        "max_episode_steps": args.max_steps,
        "num_cav": args.num_cav,
        "num_hdv": args.num_hdv,
        "num_background": args.num_background,
        "timeout": args.carla_rpc_timeout,
        "randomize_scenarios": not args.no_scenario_randomization,
    }
    env = make_env(env_config, config_path=args.config)
    mappo = MAPPOAgent.from_config(agent_config(args), state_dim=env.n_s, action_dim=env.n_a)
    ego = EgoPPOAdapter.from_config(ego_config(args), state_dim=env.n_s, action_dim=env.n_a)
    load_joint_checkpoints(args, mappo, ego)
    args._hdv_policy = load_hdv_policy(args.hdv_model) if args.hdv_model else None

    log_path = os.path.join(output_dir, "ego_train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=joint_log_fields())
        writer.writeheader()

    fixed_adv_action = None if args.resume_adv_model_dir else ACTION_NAMES[args.ego_fixed_adv_action]
    adv_label = (
        f"MAPPO:{args.resume_adv_model_dir}@{args.resume_adv_step or 'latest'}"
        if fixed_adv_action is None
        else f"fixed:{args.ego_fixed_adv_action}"
    )
    totals = {
        "adv": int(args.initial_adv_episodes),
        "ego": int(args.initial_ego_episodes),
        "global": int(args.initial_adv_episodes) + int(args.initial_ego_episodes),
    }

    print("=" * 70)
    print("Ego PPO training")
    print(f"  reward:         CatSafetyEgoReward")
    print(f"  backend:        {args.backend}")
    print(f"  config:         {args.config}")
    print(f"  episodes:       {args.episodes}")
    print(f"  max_steps:      {args.max_steps}")
    print(f"  num_cav:        {args.num_cav}")
    print(f"  adversary:      {adv_label}")
    print(f"  output_dir:     {output_dir}")
    print("=" * 70)

    try:
        train_ego_phase(
            env=env,
            mappo=mappo,
            ego=ego,
            args=args,
            round_no=0,
            min_eps=int(args.episodes),
            max_eps=int(args.episodes),
            log_path=log_path,
            model_dir=model_dir,
            totals=totals,
            fixed_adv_action=fixed_adv_action,
            log_phase="ego",
        )
        ego.save(os.path.join(model_dir, f"ego_{totals['ego']}.pt"))
        print(f"\nEgo PPO checkpoint saved under: {model_dir}")
        print(f"Ego training log: {log_path}")
    finally:
        env.close()


class AdvPolicyBuffer:
    """FIFO buffer of recent MAPPO actor/critic snapshots for ego training."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = max(int(capacity), 0)
        self.items = deque(maxlen=self.capacity)
        self.rng = np.random.RandomState(int(seed))

    @property
    def enabled(self) -> bool:
        return self.capacity > 0

    def __len__(self):
        return len(self.items)

    def snapshot(self, agent: MAPPOAgent, label: str):
        return {
            "label": str(label),
            "actor": self._clone_state_dict(agent.actor.state_dict()),
            "critic": self._clone_state_dict(agent.critic.state_dict()),
        }

    def push(self, agent: MAPPOAgent, label: str):
        if not self.enabled:
            return
        self.items.append(self.snapshot(agent, label))

    def sample(self):
        if not self.items:
            return None
        index = int(self.rng.randint(0, len(self.items)))
        return self.items[index]

    @staticmethod
    def load(agent: MAPPOAgent, snapshot):
        agent.actor.load_state_dict(snapshot["actor"])
        agent.critic.load_state_dict(snapshot["critic"])

    @staticmethod
    def _clone_state_dict(state_dict):
        cloned = {}
        for key, value in state_dict.items():
            if hasattr(value, "detach"):
                cloned[key] = value.detach().cpu().clone()
            else:
                cloned[key] = value
        return cloned


def train_ego_phase_with_adv_buffer(
    adv_policy_buffer: AdvPolicyBuffer,
    env,
    mappo: MAPPOAgent,
    ego: EgoPPOAdapter,
    args,
    *phase_args,
    **phase_kwargs,
):
    if (
        adv_policy_buffer is None
        or not adv_policy_buffer.enabled
        or len(adv_policy_buffer) == 0
        or phase_kwargs.get("fixed_adv_action") is not None
    ):
        return train_ego_phase(env, mappo, ego, args, *phase_args, **phase_kwargs)

    restore_snapshot = adv_policy_buffer.snapshot(mappo, "latest_before_ego")
    sampled_snapshot = adv_policy_buffer.sample()
    print(
        f"  [ADV-BUFFER] Ego phase uses adversarial policy: "
        f"{sampled_snapshot.get('label', 'unknown')} "
        f"(buffer={len(adv_policy_buffer)}/{adv_policy_buffer.capacity})"
    )
    AdvPolicyBuffer.load(mappo, sampled_snapshot)
    try:
        return train_ego_phase(env, mappo, ego, args, *phase_args, **phase_kwargs)
    finally:
        AdvPolicyBuffer.load(mappo, restore_snapshot)
        print("  [ADV-BUFFER] Restored latest adversarial policy after ego phase.")


def train_joint(args):
    if args.joint_strategy == "staged-pairs":
        stage_cycles = parse_stage_cycles(args.stage_cycles)
        if sum(stage_cycles) != int(args.rounds):
            raise ValueError(
                f"--rounds must equal the sum of --stage-cycles for staged-pairs: "
                f"{args.rounds} != {sum(stage_cycles)}."
            )
    if args.initial_adv_episodes < 0 or args.initial_ego_episodes < 0:
        raise ValueError("Initial episode counters cannot be negative.")

    output_dir = make_output_dir(args, prefix="joint")
    model_dir = args.model_dir or os.path.join(output_dir, "models")
    adv_model_dir = os.path.join(model_dir, "adv")
    ego_model_dir = os.path.join(model_dir, "ego")
    os.makedirs(adv_model_dir, exist_ok=True)
    os.makedirs(ego_model_dir, exist_ok=True)

    env_config = {
        "backend": args.backend,
        "max_episode_steps": args.max_steps,
        "num_cav": args.num_cav,
        "num_hdv": args.num_hdv,
        "num_background": args.num_background,
        "timeout": args.carla_rpc_timeout,
        "randomize_scenarios": not args.no_scenario_randomization,
    }
    env = make_env(env_config, config_path=args.config)
    mappo = MAPPOAgent.from_config(agent_config(args), state_dim=env.n_s, action_dim=env.n_a)
    ego = EgoPPOAdapter.from_config(ego_config(args), state_dim=env.n_s, action_dim=env.n_a)
    load_joint_checkpoints(args, mappo, ego)
    args._hdv_policy = load_hdv_policy(args.hdv_model) if args.hdv_model else None
    adv_policy_buffer = AdvPolicyBuffer(args.adv_policy_buffer_size, seed=args.seed)
    if adv_policy_buffer.enabled and args.adv_policy_buffer_include_initial:
        initial_label = f"initial_adv_{int(args.initial_adv_episodes)}"
        adv_policy_buffer.push(mappo, initial_label)

    log_path = os.path.join(output_dir, "joint_train_log.csv")
    round_log_path = os.path.join(output_dir, "joint_round_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=joint_log_fields())
        writer.writeheader()
    with open(round_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=round_log_fields())
        writer.writeheader()

    print_joint_config(args, output_dir)

    try:
        totals = {
            "adv": int(args.initial_adv_episodes),
            "ego": int(args.initial_ego_episodes),
            "global": int(args.initial_adv_episodes) + int(args.initial_ego_episodes),
        }
        if args.ego_warmup:
            print_phase_header(0, "EGO WARMUP", f"{args.ego_warmup_min_episodes}-{args.ego_warmup_max_episodes}", args.num_cav)
            warmup_info = train_ego_phase(
                env, mappo, ego, args, 0,
                args.ego_warmup_min_episodes, args.ego_warmup_max_episodes,
                log_path, ego_model_dir, totals,
                fixed_adv_action=ACTION_NAMES[args.warmup_adv_action],
                target_crash_rate=args.ego_warmup_target_crash_rate,
                target_completion=args.ego_warmup_target_completion,
                target_window=args.ego_warmup_window,
                log_phase="ego_warmup",
            )
            print_round_summary(0, empty_phase_info(), warmup_info)
            append_round_log(round_log_path, 0, empty_phase_info(), warmup_info, totals)
            ego.save(os.path.join(ego_model_dir, "ego_warmup.pt"))

        next_phase = args.start_phase
        for round_idx in range(args.rounds):
            local_round_no = round_idx + 1
            round_no = int(args.round_offset) + local_round_no
            adv_info = empty_phase_info()
            ego_info = empty_phase_info()

            if args.joint_strategy == "fixed":
                print_phase_header(round_no, "ADVERSARIAL", str(args.adv_episodes), args.num_cav)
                adv_info = train_adv_phase(
                    env, mappo, ego, args, round_no, args.adv_episodes, args.adv_episodes,
                    log_path, adv_model_dir, totals,
                )
                if adv_policy_buffer.enabled and adv_info.get("episodes", 0) > 0:
                    adv_policy_buffer.push(mappo, f"round_{round_no}_adv_{totals['adv']}")
                print_phase_header(round_no, "EGO", str(args.ego_episodes), args.num_cav)
                ego_info = train_ego_phase_with_adv_buffer(
                    adv_policy_buffer,
                    env, mappo, ego, args, round_no, args.ego_episodes, args.ego_episodes,
                    log_path, ego_model_dir, totals,
                )
            elif args.joint_strategy == "staged-pairs":
                adv_min, adv_max, ego_min, ego_max = staged_pair_eps(args, local_round_no)
                print_phase_header(round_no, "ADVERSARIAL", f"{adv_min}-{adv_max}", args.num_cav)
                adv_info = train_adv_phase(
                    env, mappo, ego, args, round_no, adv_min, adv_max,
                    log_path, adv_model_dir, totals,
                    ego_window=args.ego_window,
                    ego_improvement_threshold=args.ego_improvement_threshold,
                )
                if adv_policy_buffer.enabled and adv_info.get("episodes", 0) > 0:
                    adv_policy_buffer.push(mappo, f"round_{round_no}_adv_{totals['adv']}")
                print_phase_header(round_no, "EGO", f"{ego_min}-{ego_max}", args.num_cav)
                ego_info = train_ego_phase_with_adv_buffer(
                    adv_policy_buffer,
                    env, mappo, ego, args, round_no, ego_min, ego_max,
                    log_path, ego_model_dir, totals,
                    adv_window=args.adv_window,
                    adv_crash_threshold=args.adv_crash_threshold,
                )
            elif next_phase == "adv":
                adv_min, adv_max, _, _ = get_round_eps(args, round_no)
                print_phase_header(round_no, "ADVERSARIAL", f"{adv_min}-{adv_max}", args.num_cav)
                adv_info = train_adv_phase(
                    env, mappo, ego, args, round_no, adv_min, adv_max,
                    log_path, adv_model_dir, totals,
                    ego_window=args.ego_window,
                    ego_improvement_threshold=args.ego_improvement_threshold,
                )
                if adv_policy_buffer.enabled and adv_info.get("episodes", 0) > 0:
                    adv_policy_buffer.push(mappo, f"round_{round_no}_adv_{totals['adv']}")
                next_phase = "ego"
            else:
                _, _, ego_min, ego_max = get_round_eps(args, round_no)
                print_phase_header(round_no, "EGO", f"{ego_min}-{ego_max}", args.num_cav)
                ego_info = train_ego_phase_with_adv_buffer(
                    adv_policy_buffer,
                    env, mappo, ego, args, round_no, ego_min, ego_max,
                    log_path, ego_model_dir, totals,
                    adv_window=args.adv_window,
                    adv_crash_threshold=args.adv_crash_threshold,
                )
                next_phase = "adv"

            if args.save_interval > 0 and round_no % args.save_interval == 0:
                mappo.save(adv_model_dir, totals["adv"])
                ego.save(os.path.join(ego_model_dir, f"ego_round_{round_no}.pt"))

            print_round_summary(round_no, adv_info, ego_info)
            append_round_log(round_log_path, round_no, adv_info, ego_info, totals)

        mappo.save(adv_model_dir, totals["adv"])
        ego.save(os.path.join(ego_model_dir, f"ego_{totals['ego']}.pt"))
        print(f"\n{'=' * 70}")
        print("  TRAINING COMPLETE")
        print(f"  Results saved to: {output_dir}")
        print(f"{'=' * 70}")
    finally:
        env.close()


def train_adv_phase(
    env,
    mappo: MAPPOAgent,
    ego: EgoPPOAdapter,
    args,
    round_no: int,
    min_eps: int,
    max_eps: int,
    log_path: str,
    model_dir: str,
    totals: dict,
    ego_window: int = None,
    ego_improvement_threshold: float = None,
):
    records = []
    ego_safety_history = []
    losses = {}
    for phase_ep in range(1, max_eps + 1):
        metrics = run_adv_episode_with_ego_policy(
            env=env,
            agent=mappo,
            ego=ego,
            args=args,
            seed=episode_seed(args, totals["global"]),
            max_steps=args.max_steps,
            reward_type=args.reward_type,
        )
        losses = mappo.train()
        records.append(metrics)
        totals["adv"] += 1
        totals["global"] += 1

        survival_ratio = metrics["steps"] / max(args.max_steps, 1)
        ego_safety_history.append(survival_ratio * (1.0 - float(metrics["crash"])))

        if args.eval_interval > 0 and totals["adv"] % args.eval_interval == 0:
            summary = summarize_phase(records[-args.eval_interval:])
            print(
                f"  [ADV] Ep {totals['adv']:5d} | "
                f"Adv Reward: {summary['avg_reward']:8.2f} | "
                f"Ego Crash: {summary['crash_rate']:.1%}"
            )
            append_joint_log(log_path, joint_row(round_no, "adv", totals["adv"], summary, losses))

        if args.save_interval > 0 and totals["adv"] % args.save_interval == 0:
            mappo.save(model_dir, totals["adv"])

        if ego_window and phase_ep >= min_eps and len(ego_safety_history) >= ego_window:
            recent = ego_safety_history[-ego_window:]
            older = ego_safety_history[-2 * ego_window:-ego_window] if len(ego_safety_history) >= 2 * ego_window else [0.0] * ego_window
            improvement = float(np.mean(recent) - np.mean(older))
            if improvement < float(ego_improvement_threshold):
                print(
                    f"\n  [ADV] Early exit at ep {phase_ep}: "
                    f"ego safety plateaued (improvement={improvement:.4f} < {ego_improvement_threshold})"
                )
                break

    return summarize_phase(records, episodes=len(records))


def train_ego_phase(
    env,
    mappo: MAPPOAgent,
    ego: EgoPPOAdapter,
    args,
    round_no: int,
    min_eps: int,
    max_eps: int,
    log_path: str,
    model_dir: str,
    totals: dict,
    adv_window: int = None,
    adv_crash_threshold: float = None,
    fixed_adv_action: int = None,
    target_crash_rate: float = None,
    target_completion: float = None,
    target_window: int = None,
    log_phase: str = "ego",
):
    records = []
    adv_crash_window_values = []
    steps_since_update = 0
    losses = {}
    for phase_ep in range(1, max_eps + 1):
        metrics = run_ego_episode(
            env=env,
            mappo=mappo,
            ego=ego,
            args=args,
            seed=episode_seed(args, totals["global"]),
            max_steps=args.max_steps,
            fixed_adv_action=fixed_adv_action,
        )
        steps_since_update += int(metrics["steps"])
        if steps_since_update >= args.ppo_update_interval:
            losses = ego.update()
            steps_since_update = 0
            print_ego_update(losses)
        ego.end_episode()

        records.append(metrics)
        totals["ego"] += 1
        totals["global"] += 1

        adv_crash_window_values.append(float(metrics.get("agents_crash", False)))
        if adv_window and len(adv_crash_window_values) > adv_window:
            adv_crash_window_values.pop(0)

        if args.eval_interval > 0 and totals["ego"] % args.eval_interval == 0:
            summary = summarize_phase(records[-args.eval_interval:])
            print(
                f"  [EGO] Ep {totals['ego']:5d} | "
                f"Reward: {summary['avg_reward']:8.2f} | "
                f"Crash: {summary['crash_rate']:.1%} | "
                f"Road Compl: {summary['road_completion_rate']:.1%} | "
                f"Avg Speed: {summary['avg_speed']:.1f} m/s"
            )
            append_joint_log(log_path, joint_row(round_no, log_phase, totals["ego"], summary, losses))

        if args.save_interval > 0 and totals["ego"] % args.save_interval == 0:
            ego.save(os.path.join(model_dir, f"checkpoint-{totals['ego']}.pt"))

        if target_window and phase_ep >= min_eps and len(records) >= target_window:
            target_summary = summarize_phase(records[-target_window:])
            if (
                target_summary["crash_rate"] <= float(target_crash_rate)
                and target_summary["road_completion_rate"] >= float(target_completion)
            ):
                print(
                    f"\n  [EGO WARMUP] Target reached at ep {phase_ep}: "
                    f"crash={target_summary['crash_rate']:.1%} <= {target_crash_rate:.1%}, "
                    f"road={target_summary['road_completion_rate']:.1%} >= {target_completion:.1%}"
                )
                break

        if adv_window and phase_ep >= min_eps and len(adv_crash_window_values) >= adv_window:
            avg_adv_crash = float(np.mean(adv_crash_window_values))
            if avg_adv_crash < float(adv_crash_threshold):
                print(
                    f"\n  [EGO] Early exit at ep {phase_ep}: "
                    f"adv crash rate too low ({avg_adv_crash:.1%} < {adv_crash_threshold:.0%})"
                )
                break

    if steps_since_update > 0:
        losses = ego.update()
        print_ego_update(losses)
    return summarize_phase(records, episodes=len(records))


def load_hdv_policy(model_path: str):
    if not model_path:
        return None
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required to load --hdv-model.") from exc
    return PPO.load(model_path)


def hdv_actions(env, hdv_policy, fixed_action: int):
    observations = list(getattr(env, "obs_hdv_list", []))
    if not observations:
        return []
    if hdv_policy is None:
        return [int(fixed_action)] * len(observations)
    actions = []
    for obs in observations:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)[:25]
        action, _ = hdv_policy.predict(obs, deterministic=True)
        actions.append(int(action))
    return actions


def append_hdv_actions(env, base_actions, args):
    if str(getattr(args, "hdv_control_mode", "")).lower() in {"traffic_manager", "autopilot"}:
        return list(base_actions)
    return list(base_actions) + hdv_actions(
        env,
        getattr(args, "_hdv_policy", None),
        ACTION_NAMES[getattr(args, "hdv_action", "keep_lane")],
    )


def run_adv_episode_with_ego_policy(
    env,
    agent: MAPPOAgent,
    ego: EgoPPOAdapter,
    args,
    seed: int,
    max_steps: int,
    reward_type: str,
):
    state, reset_info = env.reset(seed=seed)
    n_agents = len(env.controlled_vehicles)
    if n_agents <= 0:
        raise RuntimeError("Joint adversarial phase requires at least one controlled CAV.")

    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    next_states = []
    active_masks = []
    agent_active = np.ones(n_agents, dtype=np.float32)
    episode_reward = 0.0
    last_info = {"route_completion": reset_info.get("route_completion", 0.0)}
    episode_diagnostics = init_episode_diagnostics()

    for step in range(max_steps):
        adv_actions, adv_log_probs, adv_values = agent.sample_actions(state, n_agents, deterministic=False)
        ego_state = np.asarray(env.obs2, dtype=np.float32).flatten()
        ego_action, _, _ = ego.select_action(ego_state, deterministic=True)
        full_action = append_hdv_actions(env, list(adv_actions) + [ego_action], args)
        next_state, global_reward, terminated, truncated, info = env.step(full_action)
        reward_vector = np.asarray(
            select_agent_rewards(info, global_reward, n_agents, reward_type), dtype=np.float32
        )
        reward_vector *= agent_active
        agent_dones = np.asarray(info.get("agents_dones", [False] * n_agents), dtype=bool)
        if agent_dones.shape != (n_agents,):
            agent_dones = np.zeros(n_agents, dtype=bool)
        transition_dones = np.logical_or(agent_dones, bool(terminated)).astype(np.float32)

        states.append(np.asarray(state, dtype=np.float32))
        actions.append(np.asarray(adv_actions, dtype=np.int64))
        rewards.append(reward_vector)
        log_probs.append(adv_log_probs)
        values.append(adv_values)
        dones.append(transition_dones)
        next_states.append(np.asarray(next_state, dtype=np.float32))
        active_masks.append(agent_active.copy())
        agent_active *= 1.0 - agent_dones.astype(np.float32)
        episode_reward += float(global_reward)
        last_info = info
        update_episode_diagnostics(episode_diagnostics, info)
        state = next_state
        agent.n_steps += 1
        if terminated or truncated:
            break

    agent.store_episode(
        states, actions, rewards, log_probs=log_probs, values=values, dones=dones,
        next_states=next_states, active_masks=active_masks,
    )
    reward_array = np.asarray(rewards, dtype=np.float32) if rewards else np.zeros((1, n_agents), dtype=np.float32)
    return episode_metrics(len(states), episode_reward, reward_array, merge_episode_diagnostics(last_info, episode_diagnostics))


def run_ego_episode(env, mappo: MAPPOAgent, ego: EgoPPOAdapter, args, seed: int, max_steps: int, fixed_adv_action: int = None):
    state, reset_info = env.reset(seed=seed)
    n_agents = len(env.controlled_vehicles)
    if n_agents <= 0:
        raise RuntimeError("Joint training requires at least one controlled CAV.")

    ego_state = np.asarray(env.obs2, dtype=np.float32).flatten()
    episode_reward = 0.0
    last_info = {"route_completion": reset_info.get("route_completion", 0.0)}
    episode_diagnostics = init_episode_diagnostics()

    for step in range(max_steps):
        if fixed_adv_action is None:
            adv_actions = mappo.exploration_action(state, n_agents)
        else:
            adv_actions = [int(fixed_adv_action)] * n_agents
        ego_action, ego_log_prob, ego_value = ego.select_action(ego_state, deterministic=False)
        full_action = append_hdv_actions(env, list(adv_actions) + [ego_action], args)
        next_state, global_reward, terminated, truncated, info = env.step(full_action)
        next_ego_state = np.asarray(env.obs2, dtype=np.float32).flatten()
        ego_reward = float(info.get("ego_reward", global_reward))
        done = bool(terminated or truncated)
        ego.step(ego_state, ego_action, ego_log_prob, ego_value, ego_reward, done, next_ego_state)
        episode_reward += ego_reward
        last_info = info
        update_episode_diagnostics(episode_diagnostics, info)
        state = next_state
        ego_state = next_ego_state
        if done:
            break

    reward_array = np.asarray(last_info.get("agents_rewards", [0.0]), dtype=np.float32)
    return episode_metrics(step + 1, episode_reward, reward_array, merge_episode_diagnostics(last_info, episode_diagnostics))


def run_adv_episode(env, agent: MAPPOAgent, ego_action: int, seed: int, max_steps: int, reward_type: str):
    state, reset_info = env.reset(seed=seed)
    n_agents = len(env.controlled_vehicles)
    if n_agents <= 0:
        raise RuntimeError("Adversarial training requires at least one controlled CAV.")

    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    next_states = []
    active_masks = []
    agent_active = np.ones(n_agents, dtype=np.float32)
    episode_reward = 0.0
    last_info = {"route_completion": reset_info.get("route_completion", 0.0)}
    episode_diagnostics = init_episode_diagnostics()

    for step in range(max_steps):
        adv_actions, adv_log_probs, adv_values = agent.sample_actions(state, n_agents, deterministic=False)
        full_action = list(adv_actions) + [ego_action]
        next_state, global_reward, terminated, truncated, info = env.step(full_action)
        reward_vector = np.asarray(
            select_agent_rewards(info, global_reward, n_agents, reward_type), dtype=np.float32
        )
        reward_vector *= agent_active
        agent_dones = np.asarray(info.get("agents_dones", [False] * n_agents), dtype=bool)
        if agent_dones.shape != (n_agents,):
            agent_dones = np.zeros(n_agents, dtype=bool)
        transition_dones = np.logical_or(agent_dones, bool(terminated)).astype(np.float32)

        states.append(np.asarray(state, dtype=np.float32))
        actions.append(np.asarray(adv_actions, dtype=np.int64))
        rewards.append(reward_vector)
        log_probs.append(adv_log_probs)
        values.append(adv_values)
        dones.append(transition_dones)
        next_states.append(np.asarray(next_state, dtype=np.float32))
        active_masks.append(agent_active.copy())
        agent_active *= 1.0 - agent_dones.astype(np.float32)
        episode_reward += float(global_reward)
        last_info = info
        update_episode_diagnostics(episode_diagnostics, info)
        state = next_state
        agent.n_steps += 1
        if terminated or truncated:
            break

    agent.store_episode(
        states, actions, rewards, log_probs=log_probs, values=values, dones=dones,
        next_states=next_states, active_masks=active_masks,
    )
    reward_array = np.asarray(rewards, dtype=np.float32) if rewards else np.zeros((1, n_agents), dtype=np.float32)
    return episode_metrics(len(states), episode_reward, reward_array, merge_episode_diagnostics(last_info, episode_diagnostics))


def episode_metrics(steps: int, episode_reward: float, reward_array, info: dict):
    agents_crash = int(info.get("adv_collision_event_count", 0)) > 0
    return {
        "steps": int(steps),
        "episode_reward": float(episode_reward),
        "mean_agent_reward": float(np.mean(reward_array)) if np.size(reward_array) else 0.0,
        "route_completion": float(info.get("route_completion", 0.0)),
        "route_completion_full": float(info.get("route_completion_full", info.get("route_completion", 0.0))),
        "route_completion_distance": float(info.get("route_completion_distance", 0.0)),
        "route_length": float(info.get("route_length", 0.0)),
        "crash": bool(info.get("crash", False)),
        "agents_crash": bool(agents_crash),
        "adv_adv_collision": int(info.get("adv_adv_collision_event_count", 0)) > 0,
        "adv_ego_collision": int(info.get("adv_ego_collision_event_count", 0)) > 0,
        "adv_other_collision": int(info.get("adv_other_collision_event_count", 0)) > 0,
        "adv_recovered": int(info.get("adv_recovery_count", 0)) > 0,
        "timeout": bool(info.get("timeout", False)),
        "average_speed": float(info.get("average_speed", 0.0)),
        "ego_target_speed": float(info.get("ego_target_speed", -1.0)),
        "ego_front_gap": float(info.get("ego_front_gap", -1.0)),
        "ego_last_action": int(info.get("ego_last_action", -1)),
        "ego_lane_change_failed": bool(info.get("ego_lane_change_failed", False)),
        "ego_lane_change_success": bool(info.get("ego_lane_change_success", False)),
        "ego_lane_change_cooldown": int(info.get("ego_lane_change_cooldown", 0)),
        "ego_stop_penalty_value": float(info.get("ego_stop_penalty_value", 0.0)),
        "ego_route_s": float(info.get("ego_route_s", 0.0)),
        "ego_lateral_offset": float(info.get("ego_lateral_offset", 0.0)),
        "ego_heading_error": float(info.get("ego_heading_error", 0.0)),
        "ego_road_id": int(info.get("ego_road_id", 0)),
        "ego_section_id": int(info.get("ego_section_id", 0)),
        "ego_raw_lane_id": int(info.get("ego_raw_lane_id", 0)),
        "ego_obstacle_distance": float(info.get("ego_obstacle_distance", -1.0)),
        "ego_obstacle_lateral": float(info.get("ego_obstacle_lateral", 0.0)),
        "ego_obstacle_risk": float(info.get("ego_obstacle_risk", 0.0)),
        "ego_obstacle_penalty": float(info.get("ego_obstacle_penalty", 0.0)),
        "ego_obstacle_escape_bonus": float(info.get("ego_obstacle_escape_bonus", 0.0)),
        "ego_obstacle_recovery_bonus": float(info.get("ego_obstacle_recovery_bonus", 0.0)),
        "ego_obstacle_takeover_penalty": float(info.get("ego_obstacle_takeover_penalty_value", 0.0)),
        "ego_obstacle_takeover_release_bonus": float(info.get("ego_obstacle_takeover_release_bonus_value", 0.0)),
        "ego_obstacle_recovered": bool(info.get("ego_obstacle_recovered", False)),
        "ego_obstacle_low_speed_pending": bool(info.get("ego_obstacle_low_speed_pending", False)),
        "ego_obstacle_distance_control": float(info.get("ego_obstacle_distance_control", -1.0)),
        "ego_obstacle_risk_control": float(info.get("ego_obstacle_risk_control", 0.0)),
        "ego_obstacle_label": str(info.get("ego_obstacle_label", "")),
        "ego_obstacle_lane_lateral": float(info.get("ego_obstacle_lane_lateral", 0.0)),
        "ego_obstacle_escape_available": bool(info.get("ego_obstacle_escape_available", False)),
        "ego_obstacle_avoidance_active": bool(info.get("ego_obstacle_avoidance_active", False)),
        "ego_obstacle_avoidance_blocked_reason": str(info.get("ego_obstacle_avoidance_blocked_reason", "")),
        "ego_obstacle_speed_limited": bool(info.get("ego_obstacle_speed_limited", False)),
        "ego_obstacle_stop_active": bool(info.get("ego_obstacle_stop_active", False)),
        "ego_obstacle_takeover_active": bool(info.get("ego_obstacle_takeover_active", False)),
        "ego_obstacle_takeover_timer": int(info.get("ego_obstacle_takeover_timer", 0)),
        "ego_obstacle_takeover_clear_steps": int(info.get("ego_obstacle_takeover_clear_steps", 0)),
        "ego_obstacle_takeover_released": bool(info.get("ego_obstacle_takeover_released", False)),
        "ego_obstacle_takeover_target_lane_id": int(info.get("ego_obstacle_takeover_target_lane_id", -999)),
        "ego_clear_path_recovery_active": bool(info.get("ego_clear_path_recovery_active", False)),
        "ego_target_lane_obstacle_risk": float(info.get("ego_target_lane_obstacle_risk", 0.0)),
        "ego_lane_obstacle_veto": bool(info.get("ego_lane_obstacle_veto", False)),
        "ego_lane_change_cancelled_for_obstacle": bool(info.get("ego_lane_change_cancelled_for_obstacle", False)),
        "ego_hazardous_lane_change": bool(info.get("ego_hazardous_lane_change", False)),
        "ego_hazardous_lane_change_risk": float(info.get("ego_hazardous_lane_change_risk", 0.0)),
        "ego_route_takeover_active": bool(info.get("ego_route_takeover_active", False)),
        "ego_route_takeover_reason": str(info.get("ego_route_takeover_reason", "")),
        "ego_route_current_error_deg": float(info.get("ego_route_current_error_deg", 0.0)),
        "ego_route_target_error_deg": float(info.get("ego_route_target_error_deg", 0.0)),
        "ego_hazardous_lane_change_penalty": float(info.get("ego_hazardous_lane_change_penalty", 0.0)),
        "ego_obstacle_risk_control_max": float(info.get("ego_obstacle_risk_control_max", 0.0)),
        "ego_target_lane_obstacle_risk_max": float(info.get("ego_target_lane_obstacle_risk_max", 0.0)),
        "ego_obstacle_risk_steps": int(info.get("ego_obstacle_risk_steps", 0)),
        "ego_obstacle_escape_available_steps": int(info.get("ego_obstacle_escape_available_steps", 0)),
        "ego_obstacle_avoidance_active_steps": int(info.get("ego_obstacle_avoidance_active_steps", 0)),
        "ego_obstacle_speed_limited_steps": int(info.get("ego_obstacle_speed_limited_steps", 0)),
        "ego_obstacle_stop_active_steps": int(info.get("ego_obstacle_stop_active_steps", 0)),
        "ego_obstacle_takeover_active_steps": int(info.get("ego_obstacle_takeover_active_steps", 0)),
        "ego_obstacle_takeover_released_steps": int(info.get("ego_obstacle_takeover_released_steps", 0)),
        "ego_lane_obstacle_veto_steps": int(info.get("ego_lane_obstacle_veto_steps", 0)),
        "ego_lane_change_cancelled_for_obstacle_steps": int(info.get("ego_lane_change_cancelled_for_obstacle_steps", 0)),
        "ego_hazardous_lane_change_steps": int(info.get("ego_hazardous_lane_change_steps", 0)),
        "ego_route_takeover_active_steps": int(info.get("ego_route_takeover_active_steps", 0)),
        "ego_clear_path_recovery_active_steps": int(info.get("ego_clear_path_recovery_active_steps", 0)),
        "ego_hazardous_lane_change_penalty_sum": float(info.get("ego_hazardous_lane_change_penalty_sum", 0.0)),
        "ego_obstacle_risk_any": bool(info.get("ego_obstacle_risk_any", False)),
        "ego_obstacle_escape_available_any": bool(info.get("ego_obstacle_escape_available_any", False)),
        "ego_obstacle_avoidance_active_any": bool(info.get("ego_obstacle_avoidance_active_any", False)),
        "ego_obstacle_speed_limited_any": bool(info.get("ego_obstacle_speed_limited_any", False)),
        "ego_obstacle_stop_active_any": bool(info.get("ego_obstacle_stop_active_any", False)),
        "ego_obstacle_takeover_active_any": bool(info.get("ego_obstacle_takeover_active_any", False)),
        "ego_obstacle_takeover_released_any": bool(info.get("ego_obstacle_takeover_released_any", False)),
        "ego_lane_obstacle_veto_any": bool(info.get("ego_lane_obstacle_veto_any", False)),
        "ego_lane_change_cancelled_for_obstacle_any": bool(info.get("ego_lane_change_cancelled_for_obstacle_any", False)),
        "ego_hazardous_lane_change_any": bool(info.get("ego_hazardous_lane_change_any", False)),
        "ego_route_takeover_active_any": bool(info.get("ego_route_takeover_active_any", False)),
        "ego_clear_path_recovery_active_any": bool(info.get("ego_clear_path_recovery_active_any", False)),
        "ego_risk_raw": float(info.get("ego_risk_raw", info.get("ego_risk_value", 0.0))),
        "ego_adaptive_tau": float(info.get("ego_adaptive_tau", 0.0)),
        "ego_risk_violation": float(info.get("ego_risk_violation", 0.0)),
        "ego_risk_constraint_penalty": float(info.get("ego_risk_constraint_penalty", 0.0)),
        "ego_interaction_entropy": float(info.get("ego_interaction_entropy", 0.0)),
        "ego_interaction_coverage": float(info.get("ego_interaction_coverage", 0.0)),
        "ego_performance_easing": float(info.get("ego_performance_easing", 0.0)),
        "adv_ego_behavior_change": float(info.get("adv_ego_behavior_change", 0.0)),
        "adv_ego_speed_change": float(info.get("adv_ego_speed_change", 0.0)),
        "adv_ego_deceleration": float(info.get("adv_ego_deceleration", 0.0)),
        "adv_ego_lane_change": float(info.get("adv_ego_lane_change", 0.0)),
        "adv_interaction_variance": float(info.get("adv_interaction_variance", 0.0)),
        "adv_interaction_coverage": float(info.get("adv_interaction_coverage", info.get("adv_interaction_variance", 0.0))),
        "adv_interaction_entropy": float(info.get("adv_interaction_entropy", 0.0)),
        "adv_diversity_reward": float(info.get("adv_diversity_reward", 0.0)),
        "adv_mean_close_distance_penalty": float(np.mean(info.get("adv_close_distance_penalties", (0.0,)))) if info.get("adv_close_distance_penalties", ()) else 0.0,
        "adv_mean_ttc_penalty": float(np.mean(info.get("adv_ttc_penalties", (0.0,)))) if info.get("adv_ttc_penalties", ()) else 0.0,
        "adv_mean_duplicate_mode_penalty": float(np.mean(info.get("adv_duplicate_mode_penalties", (0.0,)))) if info.get("adv_duplicate_mode_penalties", ()) else 0.0,
        "adv_min_team_distance": min((value for value in info.get("adv_min_team_distances", ()) if value >= 0.0), default=-1.0),
        "adv_min_team_ttc": min((value for value in info.get("adv_min_team_ttcs", ()) if value >= 0.0), default=-1.0),
        "active_adv_count": int(info.get("active_adv_count", info.get("adv_active_count", 0))),
        "newly_collided_non_ego_count": int(info.get("newly_collided_non_ego_count", 0)),
        "adv_collision_event_count": int(info.get("adv_collision_event_count", 0)),
        "adv_adv_collision_event_count": int(info.get("adv_adv_collision_event_count", 0)),
        "adv_ego_collision_event_count": int(info.get("adv_ego_collision_event_count", 0)),
        "adv_other_collision_event_count": int(info.get("adv_other_collision_event_count", 0)),
        "adv_recovery_count": int(info.get("adv_recovery_count", 0)),
        "policy_inactive_non_ego_count": int(info.get("policy_inactive_non_ego_count", 0)),
        "physical_accident_vehicle_count": int(info.get("physical_accident_vehicle_count", 0)),
        "recovering_adv_count": int(info.get("recovering_adv_count", 0)),
        "adv_front_safety_active_count": int(info.get("adv_front_safety_active_count", 0)),
        "adv_front_safety_mean_brake": float(info.get("adv_front_safety_mean_brake", 0.0)),
        "adv_front_control_min_gap": float(info.get("adv_front_control_min_gap", -1.0)),
        "adv_front_control_mean_speed_limit": float(info.get("adv_front_control_mean_speed_limit", -1.0)),
        "adv_mean_speed_match_reward": float(np.mean(info.get("adv_speed_match_rewards", (0.0,)))) if info.get("adv_speed_match_rewards", ()) else 0.0,
    }


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def init_episode_diagnostics():
    return {
        "ego_obstacle_risk_control_max": 0.0,
        "ego_target_lane_obstacle_risk_max": 0.0,
        "ego_obstacle_risk_steps": 0,
        "ego_obstacle_escape_available_steps": 0,
        "ego_obstacle_avoidance_active_steps": 0,
        "ego_obstacle_speed_limited_steps": 0,
        "ego_obstacle_stop_active_steps": 0,
        "ego_obstacle_takeover_active_steps": 0,
        "ego_obstacle_takeover_released_steps": 0,
        "ego_lane_obstacle_veto_steps": 0,
        "ego_lane_change_cancelled_for_obstacle_steps": 0,
        "ego_hazardous_lane_change_steps": 0,
        "ego_route_takeover_active_steps": 0,
        "ego_clear_path_recovery_active_steps": 0,
        "ego_hazardous_lane_change_penalty_sum": 0.0,
    }


def update_episode_diagnostics(diagnostics, info):
    obstacle_risk = _as_float(info.get("ego_obstacle_risk_control", 0.0))
    target_risk = _as_float(info.get("ego_target_lane_obstacle_risk", 0.0))
    diagnostics["ego_obstacle_risk_control_max"] = max(
        diagnostics["ego_obstacle_risk_control_max"], obstacle_risk
    )
    diagnostics["ego_target_lane_obstacle_risk_max"] = max(
        diagnostics["ego_target_lane_obstacle_risk_max"], target_risk
    )
    diagnostics["ego_obstacle_risk_steps"] += int(obstacle_risk > 0.0 or target_risk > 0.0)
    diagnostics["ego_obstacle_escape_available_steps"] += int(_as_bool(info.get("ego_obstacle_escape_available", False)))
    diagnostics["ego_obstacle_avoidance_active_steps"] += int(_as_bool(info.get("ego_obstacle_avoidance_active", False)))
    diagnostics["ego_obstacle_speed_limited_steps"] += int(_as_bool(info.get("ego_obstacle_speed_limited", False)))
    diagnostics["ego_obstacle_stop_active_steps"] += int(_as_bool(info.get("ego_obstacle_stop_active", False)))
    diagnostics["ego_obstacle_takeover_active_steps"] += int(_as_bool(info.get("ego_obstacle_takeover_active", False)))
    diagnostics["ego_obstacle_takeover_released_steps"] += int(_as_bool(info.get("ego_obstacle_takeover_released", False)))
    diagnostics["ego_lane_obstacle_veto_steps"] += int(_as_bool(info.get("ego_lane_obstacle_veto", False)))
    diagnostics["ego_lane_change_cancelled_for_obstacle_steps"] += int(
        _as_bool(info.get("ego_lane_change_cancelled_for_obstacle", False))
    )
    diagnostics["ego_hazardous_lane_change_steps"] += int(_as_bool(info.get("ego_hazardous_lane_change", False)))
    diagnostics["ego_route_takeover_active_steps"] += int(_as_bool(info.get("ego_route_takeover_active", False)))
    diagnostics["ego_clear_path_recovery_active_steps"] += int(_as_bool(info.get("ego_clear_path_recovery_active", False)))
    diagnostics["ego_hazardous_lane_change_penalty_sum"] += _as_float(
        info.get("ego_hazardous_lane_change_penalty", 0.0)
    )


def merge_episode_diagnostics(info, diagnostics):
    merged = dict(info)
    merged.update(diagnostics)
    merged["ego_obstacle_risk_any"] = bool(diagnostics["ego_obstacle_risk_steps"] > 0)
    merged["ego_obstacle_escape_available_any"] = bool(diagnostics["ego_obstacle_escape_available_steps"] > 0)
    merged["ego_obstacle_avoidance_active_any"] = bool(diagnostics["ego_obstacle_avoidance_active_steps"] > 0)
    merged["ego_obstacle_speed_limited_any"] = bool(diagnostics["ego_obstacle_speed_limited_steps"] > 0)
    merged["ego_obstacle_stop_active_any"] = bool(diagnostics["ego_obstacle_stop_active_steps"] > 0)
    merged["ego_obstacle_takeover_active_any"] = bool(diagnostics["ego_obstacle_takeover_active_steps"] > 0)
    merged["ego_obstacle_takeover_released_any"] = bool(diagnostics["ego_obstacle_takeover_released_steps"] > 0)
    merged["ego_lane_obstacle_veto_any"] = bool(diagnostics["ego_lane_obstacle_veto_steps"] > 0)
    merged["ego_lane_change_cancelled_for_obstacle_any"] = bool(
        diagnostics["ego_lane_change_cancelled_for_obstacle_steps"] > 0
    )
    merged["ego_hazardous_lane_change_any"] = bool(diagnostics["ego_hazardous_lane_change_steps"] > 0)
    merged["ego_route_takeover_active_any"] = bool(diagnostics["ego_route_takeover_active_steps"] > 0)
    merged["ego_clear_path_recovery_active_any"] = bool(diagnostics["ego_clear_path_recovery_active_steps"] > 0)
    return merged


def select_agent_rewards(info, global_reward: float, n_agents: int, reward_type: str):
    if reward_type == "regionalR":
        values = info.get("regional_rewards", ())
    elif reward_type == "agents_rewards":
        values = info.get("agents_rewards", ())
    else:
        values = [global_reward] * n_agents
    if len(values) != n_agents:
        values = [global_reward] * n_agents
    return [float(value) for value in values]


def index_to_one_hot_safe(index: int, dim: int):
    one_hot = np.zeros(dim, dtype=np.float32)
    one_hot[int(index)] = 1.0
    return one_hot


def make_output_dir(args, prefix="adv"):
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = os.path.join(args.base_dir, prefix + "_" + now)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def agent_config(args):
    return {
        "memory_capacity": args.memory_capacity,
        "batch_size": args.batch_size,
        "reward_gamma": args.reward_gamma,
        "reward_scale": args.reward_scale,
        "actor_hidden_size": args.actor_hidden_size,
        "critic_hidden_size": args.critic_hidden_size,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "entropy_reg": args.entropy_reg,
        "value_coef": args.value_coef,
        "gae_lambda": args.gae_lambda,
        "ppo_epochs": args.mappo_ppo_epochs,
        "normalize_advantages": True,
        "max_grad_norm": args.max_grad_norm,
        "target_update_steps": args.target_update_steps,
        "target_tau": args.target_tau,
        "clip_param": args.clip_param,
        "reward_type": args.reward_type,
        "episodes_before_train": args.episodes_before_train,
        "use_cuda": not args.no_cuda,
        "torch_seed": args.torch_seed,
    }


def ego_config(args):
    return {
        "n_steps": args.ego_n_steps,
        "batch_size": args.ego_batch_size,
        "actor_lr": args.ego_lr,
        "critic_lr": args.ego_lr,
        "use_cuda": not args.no_cuda,
        "seed": args.torch_seed,
    }


def get_round_eps(args, round_no: int):
    if args.no_progressive:
        return args.adv_min_eps, args.adv_max_eps, args.ego_min_eps, args.ego_max_eps
    return progressive_eps(round_no, args.rounds)


def parse_stage_cycles(value: str):
    try:
        cycles = tuple(int(item.strip()) for item in str(value).split(","))
    except ValueError as exc:
        raise ValueError("--stage-cycles must contain three comma-separated integers.") from exc
    if len(cycles) != 3 or any(item < 0 for item in cycles) or sum(cycles) <= 0:
        raise ValueError("--stage-cycles must contain three non-negative integers with a positive sum.")
    return cycles


def staged_pair_eps(args, local_round_no: int):
    initial_cycles, middle_cycles, final_cycles = parse_stage_cycles(args.stage_cycles)
    total_cycles = initial_cycles + middle_cycles + final_cycles
    if int(args.rounds) != total_cycles:
        raise ValueError(
            f"staged-pairs requires --rounds {total_cycles} for "
            f"--stage-cycles {args.stage_cycles}, got {args.rounds}."
        )
    if local_round_no <= initial_cycles:
        return 150, 300, 300, 500
    if local_round_no <= initial_cycles + middle_cycles:
        return 100, 200, 500, 800
    return 80, 150, 800, 1000


def load_joint_checkpoints(args, mappo: MAPPOAgent, ego: EgoPPOAdapter):
    adv_dir = args.resume_adv_model_dir
    adv_step = args.resume_adv_step
    if adv_dir is None and adv_step is not None:
        raise ValueError("--resume-adv-step requires --resume-adv-model-dir.")
    if adv_dir is not None:
        if adv_step is None:
            adv_step = latest_mappo_step(adv_dir)
            args.resume_adv_step = adv_step
        mappo.load(adv_dir, int(adv_step), train_mode=True)
        if int(args.initial_adv_episodes) == 0:
            args.initial_adv_episodes = int(adv_step)
        mappo.n_episodes = int(args.initial_adv_episodes)
        print(f"[RESUME] MAPPO: {adv_dir} step={adv_step}")

    if args.resume_ego_checkpoint:
        ego_checkpoint = args.resume_ego_checkpoint
        if os.path.isdir(ego_checkpoint):
            ego_checkpoint = latest_ego_checkpoint(ego_checkpoint)
            args.resume_ego_checkpoint = ego_checkpoint
        ego.load(ego_checkpoint)
        if int(args.initial_ego_episodes) == 0:
            args.initial_ego_episodes = int(ego.policy.n_episodes)
        print(f"[RESUME] EgoPPO: {ego_checkpoint}")


def latest_mappo_step(model_dir: str) -> int:
    actor_steps = checkpoint_steps(model_dir, r"actor_(\d+)\.pt")
    critic_steps = checkpoint_steps(model_dir, r"critic_(\d+)\.pt")
    common_steps = actor_steps & critic_steps
    if not common_steps:
        raise FileNotFoundError(f"No matching MAPPO actor/critic checkpoints found in {model_dir}.")
    return max(common_steps)


def latest_ego_checkpoint(model_dir: str) -> str:
    candidates = []
    pattern = re.compile(r"checkpoint-(\d+)\.pt$")
    for name in os.listdir(model_dir):
        match = pattern.fullmatch(name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(model_dir, name)))
    if not candidates:
        raise FileNotFoundError(f"No Ego checkpoint-N.pt files found in {model_dir}.")
    return max(candidates)[1]


def checkpoint_steps(model_dir: str, pattern: str):
    regex = re.compile(pattern)
    steps = set()
    for name in os.listdir(model_dir):
        match = regex.fullmatch(name)
        if match:
            steps.add(int(match.group(1)))
    return steps


def progressive_eps(round_idx: int, n_rounds: int):
    ratio = round_idx / max(n_rounds, 1)
    if ratio <= 10 / 70:
        return 150, 300, 300, 500
    if ratio <= 40 / 70:
        return 100, 200, 500, 800
    return 80, 150, 800, 1000


def summarize_phase(records, episodes=None):
    if not records:
        return empty_phase_info() if episodes is None else {**empty_phase_info(), "episodes": episodes}
    last_record = records[-1]
    return {
        "episodes": len(records) if episodes is None else int(episodes),
        "avg_reward": float(np.mean([r.get("episode_reward", 0.0) for r in records])),
        "mean_agent_reward": float(np.mean([r.get("mean_agent_reward", 0.0) for r in records])),
        "crash_rate": float(np.mean([float(r.get("crash", False)) for r in records])),
        "agents_crash_rate": float(np.mean([float(r.get("agents_crash", False)) for r in records])),
        "adv_adv_collision_rate": float(np.mean([float(r.get("adv_adv_collision", False)) for r in records])),
        "adv_ego_collision_rate": float(np.mean([float(r.get("adv_ego_collision", False)) for r in records])),
        "adv_other_collision_rate": float(np.mean([float(r.get("adv_other_collision", False)) for r in records])),
        "adv_recovery_rate": float(np.mean([float(r.get("adv_recovered", False)) for r in records])),
        "timeout_rate": float(np.mean([float(r.get("timeout", False)) for r in records])),
        "road_completion_rate": float(np.mean([r.get("route_completion", 0.0) for r in records])),
        "road_completion_rate_full": float(np.mean([r.get("route_completion_full", r.get("route_completion", 0.0)) for r in records])),
        "route_completion_distance": float(np.mean([r.get("route_completion_distance", 0.0) for r in records])),
        "route_length": float(np.mean([r.get("route_length", 0.0) for r in records])),
        "avg_speed": float(np.mean([r.get("average_speed", 0.0) for r in records])),
        "avg_steps": float(np.mean([r.get("steps", 0) for r in records])),
        **diagnostic_row(last_record),
        **summarize_diagnostics(records),
    }


def empty_phase_info():
    return {
        "episodes": 0,
        "avg_reward": 0.0,
        "mean_agent_reward": 0.0,
        "crash_rate": 0.0,
        "agents_crash_rate": 0.0,
        "adv_adv_collision_rate": 0.0,
        "adv_ego_collision_rate": 0.0,
        "adv_other_collision_rate": 0.0,
        "adv_recovery_rate": 0.0,
        "timeout_rate": 0.0,
        "road_completion_rate": 0.0,
        "road_completion_rate_full": 0.0,
        "route_completion_distance": 0.0,
        "route_length": 0.0,
        "avg_speed": 0.0,
        "avg_steps": 0.0,
        **diagnostic_row({}),
        **summarize_diagnostics([]),
    }


def summarize_diagnostics(records):
    count_fields = [
        "ego_obstacle_risk_steps",
        "ego_obstacle_escape_available_steps",
        "ego_obstacle_avoidance_active_steps",
        "ego_obstacle_speed_limited_steps",
        "ego_obstacle_stop_active_steps",
        "ego_obstacle_takeover_active_steps",
        "ego_obstacle_takeover_released_steps",
        "ego_lane_obstacle_veto_steps",
        "ego_lane_change_cancelled_for_obstacle_steps",
        "ego_hazardous_lane_change_steps",
        "ego_route_takeover_active_steps",
        "ego_clear_path_recovery_active_steps",
    ]
    any_fields = [
        "ego_obstacle_risk_any",
        "ego_obstacle_escape_available_any",
        "ego_obstacle_avoidance_active_any",
        "ego_obstacle_speed_limited_any",
        "ego_obstacle_stop_active_any",
        "ego_obstacle_takeover_active_any",
        "ego_obstacle_takeover_released_any",
        "ego_lane_obstacle_veto_any",
        "ego_lane_change_cancelled_for_obstacle_any",
        "ego_hazardous_lane_change_any",
        "ego_route_takeover_active_any",
        "ego_clear_path_recovery_active_any",
    ]
    max_fields = [
        "ego_obstacle_risk_control_max",
        "ego_target_lane_obstacle_risk_max",
    ]
    sum_fields = [
        "ego_hazardous_lane_change_penalty_sum",
    ]
    if not records:
        return {
            **{field: 0 for field in count_fields},
            **{field: False for field in any_fields},
            **{field: 0.0 for field in max_fields},
            **{field: 0.0 for field in sum_fields},
        }
    return {
        **{field: int(sum(_as_float(record.get(field, 0.0)) for record in records)) for field in count_fields},
        **{field: bool(any(_as_bool(record.get(field, False)) for record in records)) for field in any_fields},
        **{field: float(max(_as_float(record.get(field, 0.0)) for record in records)) for field in max_fields},
        **{field: float(sum(_as_float(record.get(field, 0.0)) for record in records)) for field in sum_fields},
    }


def diagnostic_fields():
    return [
        "ego_target_speed",
        "ego_front_gap",
        "ego_last_action",
        "ego_lane_change_failed",
        "ego_lane_change_success",
        "ego_lane_change_cooldown",
        "ego_stop_penalty_value",
        "ego_route_s",
        "ego_lateral_offset",
        "ego_heading_error",
        "ego_road_id",
        "ego_section_id",
        "ego_raw_lane_id",
        "ego_obstacle_distance",
        "ego_obstacle_lateral",
        "ego_obstacle_risk",
        "ego_obstacle_penalty",
        "ego_obstacle_escape_bonus",
        "ego_obstacle_recovery_bonus",
        "ego_obstacle_takeover_penalty",
        "ego_obstacle_takeover_release_bonus",
        "ego_obstacle_recovered",
        "ego_obstacle_low_speed_pending",
        "ego_obstacle_distance_control",
        "ego_obstacle_risk_control",
        "ego_obstacle_label",
        "ego_obstacle_lane_lateral",
        "ego_obstacle_escape_available",
        "ego_obstacle_avoidance_active",
        "ego_obstacle_avoidance_blocked_reason",
        "ego_obstacle_speed_limited",
        "ego_obstacle_stop_active",
        "ego_obstacle_takeover_active",
        "ego_obstacle_takeover_timer",
        "ego_obstacle_takeover_clear_steps",
        "ego_obstacle_takeover_released",
        "ego_obstacle_takeover_target_lane_id",
        "ego_clear_path_recovery_active",
        "ego_target_lane_obstacle_risk",
        "ego_lane_obstacle_veto",
        "ego_lane_change_cancelled_for_obstacle",
        "ego_hazardous_lane_change",
        "ego_hazardous_lane_change_risk",
        "ego_route_takeover_active",
        "ego_route_takeover_reason",
        "ego_route_current_error_deg",
        "ego_route_target_error_deg",
        "ego_hazardous_lane_change_penalty",
        "ego_obstacle_risk_control_max",
        "ego_target_lane_obstacle_risk_max",
        "ego_obstacle_risk_steps",
        "ego_obstacle_escape_available_steps",
        "ego_obstacle_avoidance_active_steps",
        "ego_obstacle_speed_limited_steps",
        "ego_obstacle_stop_active_steps",
        "ego_obstacle_takeover_active_steps",
        "ego_obstacle_takeover_released_steps",
        "ego_lane_obstacle_veto_steps",
        "ego_lane_change_cancelled_for_obstacle_steps",
        "ego_hazardous_lane_change_steps",
        "ego_route_takeover_active_steps",
        "ego_clear_path_recovery_active_steps",
        "ego_hazardous_lane_change_penalty_sum",
        "ego_obstacle_risk_any",
        "ego_obstacle_escape_available_any",
        "ego_obstacle_avoidance_active_any",
        "ego_obstacle_speed_limited_any",
        "ego_obstacle_stop_active_any",
        "ego_obstacle_takeover_active_any",
        "ego_obstacle_takeover_released_any",
        "ego_lane_obstacle_veto_any",
        "ego_lane_change_cancelled_for_obstacle_any",
        "ego_hazardous_lane_change_any",
        "ego_route_takeover_active_any",
        "ego_clear_path_recovery_active_any",
        "ego_risk_raw",
        "ego_adaptive_tau",
        "ego_risk_violation",
        "ego_risk_constraint_penalty",
        "ego_interaction_entropy",
        "ego_interaction_coverage",
        "ego_performance_easing",
        "adv_ego_behavior_change",
        "adv_ego_speed_change",
        "adv_ego_deceleration",
        "adv_ego_lane_change",
        "adv_interaction_variance",
        "adv_interaction_coverage",
        "adv_interaction_entropy",
        "adv_diversity_reward",
        "adv_mean_close_distance_penalty",
        "adv_mean_ttc_penalty",
        "adv_mean_duplicate_mode_penalty",
        "adv_min_team_distance",
        "adv_min_team_ttc",
        "active_adv_count",
        "newly_collided_non_ego_count",
        "adv_collision_event_count",
        "adv_adv_collision_event_count",
        "adv_ego_collision_event_count",
        "adv_other_collision_event_count",
        "adv_recovery_count",
        "policy_inactive_non_ego_count",
        "physical_accident_vehicle_count",
        "recovering_adv_count",
        "adv_front_safety_active_count",
        "adv_front_safety_mean_brake",
        "adv_front_control_min_gap",
        "adv_front_control_mean_speed_limit",
        "adv_mean_speed_match_reward",
    ]


def diagnostic_row(metrics):
    return {field: metrics.get(field, "") for field in diagnostic_fields()}


def log_fields():
    return [
        "episode",
        "steps",
        "episode_reward",
        "mean_agent_reward",
        "route_completion",
        "route_completion_full",
        "route_completion_distance",
        "route_length",
        "crash",
        "agents_crash",
        "timeout",
        *diagnostic_fields(),
        "actor_loss",
        "critic_loss",
        "entropy",
        "clip_fraction",
        "value_error",
        "ppo_epochs",
        "samples",
    ]


def episode_log_row(episode, metrics, losses):
    return {
        "episode": episode,
        "steps": metrics.get("steps", 0),
        "episode_reward": metrics.get("episode_reward", 0.0),
        "mean_agent_reward": metrics.get("mean_agent_reward", 0.0),
        "route_completion": metrics.get("route_completion", 0.0),
        "route_completion_full": metrics.get("route_completion_full", metrics.get("route_completion", 0.0)),
        "route_completion_distance": metrics.get("route_completion_distance", 0.0),
        "route_length": metrics.get("route_length", 0.0),
        "crash": metrics.get("crash", False),
        "agents_crash": metrics.get("agents_crash", False),
        "timeout": metrics.get("timeout", False),
        **diagnostic_row(metrics),
        "actor_loss": losses.get("actor_loss", ""),
        "critic_loss": losses.get("critic_loss", ""),
        "entropy": losses.get("entropy", ""),
        "clip_fraction": losses.get("clip_fraction", ""),
        "value_error": losses.get("value_error", ""),
        "ppo_epochs": losses.get("ppo_epochs", ""),
        "samples": losses.get("samples", ""),
    }


def append_log(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields())
        writer.writerow(row)


def joint_log_fields():
    return [
        "round",
        "phase",
        "episode",
        "episodes_in_window",
        "avg_steps",
        "episode_reward",
        "mean_agent_reward",
        "route_completion",
        "route_completion_full",
        "route_completion_distance",
        "route_length",
        "crash_rate",
        "agents_crash_rate",
        "adv_adv_collision_rate",
        "adv_ego_collision_rate",
        "adv_other_collision_rate",
        "adv_recovery_rate",
        "timeout_rate",
        *diagnostic_fields(),
        "actor_loss",
        "critic_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "clip_fraction",
        "value_error",
        "ppo_epochs",
        "samples",
    ]


def joint_row(round_idx, phase, episode, metrics, losses):
    return {
        "round": round_idx,
        "phase": phase,
        "episode": episode,
        "episodes_in_window": metrics.get("episodes", 0),
        "avg_steps": metrics.get("avg_steps", metrics.get("steps", 0)),
        "episode_reward": metrics.get("avg_reward", metrics.get("episode_reward", 0.0)),
        "mean_agent_reward": metrics.get("mean_agent_reward", 0.0),
        "route_completion": metrics.get("road_completion_rate", metrics.get("route_completion", 0.0)),
        "route_completion_full": metrics.get("road_completion_rate_full", metrics.get("route_completion_full", "")),
        "route_completion_distance": metrics.get("route_completion_distance", ""),
        "route_length": metrics.get("route_length", ""),
        "crash_rate": metrics.get("crash_rate", metrics.get("crash", False)),
        "agents_crash_rate": metrics.get("agents_crash_rate", metrics.get("agents_crash", False)),
        "adv_adv_collision_rate": metrics.get("adv_adv_collision_rate", metrics.get("adv_adv_collision", False)),
        "adv_ego_collision_rate": metrics.get("adv_ego_collision_rate", metrics.get("adv_ego_collision", False)),
        "adv_other_collision_rate": metrics.get("adv_other_collision_rate", metrics.get("adv_other_collision", False)),
        "adv_recovery_rate": metrics.get("adv_recovery_rate", metrics.get("adv_recovered", False)),
        "timeout_rate": metrics.get("timeout_rate", metrics.get("timeout", False)),
        **diagnostic_row(metrics),
        "actor_loss": losses.get("actor_loss", ""),
        "critic_loss": losses.get("critic_loss", ""),
        "policy_loss": losses.get("policy_loss", ""),
        "value_loss": losses.get("value_loss", ""),
        "entropy": losses.get("entropy", ""),
        "clip_fraction": losses.get("clip_fraction", ""),
        "value_error": losses.get("value_error", ""),
        "ppo_epochs": losses.get("ppo_epochs", ""),
        "samples": losses.get("samples", ""),
    }


def append_joint_log(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=joint_log_fields())
        writer.writerow(row)


def round_log_fields():
    return [
        "round",
        "total_adversarial_episodes",
        "total_ego_episodes",
        "adv_episodes",
        "adv_avg_reward",
        "adv_crash_rate",
        "adv_agents_crash_rate",
        "adv_adv_collision_rate",
        "adv_ego_collision_rate",
        "adv_other_collision_rate",
        "adv_recovery_rate",
        "adv_road_completion_rate",
        "adv_road_completion_rate_full",
        "ego_episodes",
        "ego_avg_reward",
        "ego_crash_rate",
        "ego_agents_crash_rate",
        "ego_adv_adv_collision_rate",
        "ego_adv_ego_collision_rate",
        "ego_adv_other_collision_rate",
        "ego_adv_recovery_rate",
        "ego_road_completion_rate",
        "ego_road_completion_rate_full",
        "ego_route_completion_distance",
        "ego_route_length",
        "ego_avg_speed",
    ]


def append_round_log(path, round_no, adv_info, ego_info, totals):
    row = {
        "round": round_no,
        "total_adversarial_episodes": totals["adv"],
        "total_ego_episodes": totals["ego"],
        "adv_episodes": adv_info.get("episodes", 0),
        "adv_avg_reward": adv_info.get("avg_reward", 0.0),
        "adv_crash_rate": adv_info.get("crash_rate", 0.0),
        "adv_agents_crash_rate": adv_info.get("agents_crash_rate", 0.0),
        "adv_adv_collision_rate": adv_info.get("adv_adv_collision_rate", 0.0),
        "adv_ego_collision_rate": adv_info.get("adv_ego_collision_rate", 0.0),
        "adv_other_collision_rate": adv_info.get("adv_other_collision_rate", 0.0),
        "adv_recovery_rate": adv_info.get("adv_recovery_rate", 0.0),
        "adv_road_completion_rate": adv_info.get("road_completion_rate", 0.0),
        "adv_road_completion_rate_full": adv_info.get("road_completion_rate_full", 0.0),
        "ego_episodes": ego_info.get("episodes", 0),
        "ego_avg_reward": ego_info.get("avg_reward", 0.0),
        "ego_crash_rate": ego_info.get("crash_rate", 0.0),
        "ego_agents_crash_rate": ego_info.get("agents_crash_rate", 0.0),
        "ego_adv_adv_collision_rate": ego_info.get("adv_adv_collision_rate", 0.0),
        "ego_adv_ego_collision_rate": ego_info.get("adv_ego_collision_rate", 0.0),
        "ego_adv_other_collision_rate": ego_info.get("adv_other_collision_rate", 0.0),
        "ego_adv_recovery_rate": ego_info.get("adv_recovery_rate", 0.0),
        "ego_road_completion_rate": ego_info.get("road_completion_rate", 0.0),
        "ego_road_completion_rate_full": ego_info.get("road_completion_rate_full", 0.0),
        "ego_route_completion_distance": ego_info.get("route_completion_distance", 0.0),
        "ego_route_length": ego_info.get("route_length", 0.0),
        "ego_avg_speed": ego_info.get("avg_speed", 0.0),
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=round_log_fields())
        writer.writerow(row)


def print_joint_config(args, output_dir):
    print("=" * 70)
    if args.joint_strategy == "adaptive":
        title = "ADAPTIVE JOINT ADVERSARIAL TRAINING"
    elif args.joint_strategy == "staged-pairs":
        title = "STAGED PAIRED JOINT ADVERSARIAL TRAINING"
    else:
        title = "JOINT ADVERSARIAL TRAINING - Phased Alternating"
    print(f"  {title}")
    print("=" * 70)
    print(f"  backend:              {args.backend}")
    print(f"  rounds:               {args.rounds}")
    print(f"  round offset:         {args.round_offset}")
    print(f"  num_cav:              {args.num_cav}")
    print(f"  num_hdv:              {args.num_hdv}")
    print(f"  num_background:       {args.num_background}")
    print(f"  max_steps:            {args.max_steps}")
    print(f"  reward_type:          {args.reward_type}")
    print(f"  MAPPO GAE lambda:     {args.gae_lambda}")
    print(f"  MAPPO PPO epochs:     {args.mappo_ppo_epochs}")
    print(f"  MAPPO entropy coef:   {args.entropy_reg}")
    print(f"  ADV policy buffer:    {args.adv_policy_buffer_size}")
    print(f"  scenario_randomize:   {not args.no_scenario_randomization}")
    if args.joint_strategy == "adaptive":
        first = args.start_phase.upper()
        second = "EGO" if args.start_phase == "adv" else "ADV"
        print(f"  phase order:          {first} -> {second} -> ...")
        if not args.no_progressive:
            print("  Episode allocation:   PROGRESSIVE")
            print("    Stage 1 (R1-R10):   ADV 150-300, EGO 300-500")
            print("    Stage 2 (R11-R40):  ADV 100-200, EGO 500-800")
            print("    Stage 3 (R41+):     ADV 80-150,  EGO 800-1000")
        else:
            print(f"  ADV phase:            {args.adv_min_eps}-{args.adv_max_eps} eps")
            print(f"  EGO phase:            {args.ego_min_eps}-{args.ego_max_eps} eps")
        print(f"  adv crash threshold:  {args.adv_crash_threshold:.0%}")
        print(f"  adv crash window:     {args.adv_window} eps")
        print(f"  ego safety window:    {args.ego_window} eps")
    elif args.joint_strategy == "staged-pairs":
        initial_cycles, middle_cycles, final_cycles = parse_stage_cycles(args.stage_cycles)
        print("  phase order:          ADV -> EGO in every round")
        print("  Episode allocation:   STAGED PAIRS")
        print(
            f"    Initial ({initial_cycles} rounds): "
            "ADV 150-300, EGO 300-500"
        )
        print(
            f"    Middle  ({middle_cycles} rounds): "
            "ADV 100-200, EGO 500-800"
        )
        print(
            f"    Final   ({final_cycles} rounds): "
            "ADV 80-150,  EGO 800-1000"
        )
    else:
        print(f"  adv episodes/round:   {args.adv_episodes}")
        print(f"  ego episodes/round:   {args.ego_episodes}")
    if args.resume_adv_model_dir:
        print(f"  resume MAPPO:         {args.resume_adv_model_dir} @ {args.resume_adv_step}")
    if args.resume_ego_checkpoint:
        print(f"  resume EgoPPO:        {args.resume_ego_checkpoint}")
    if args.hdv_model:
        print(f"  frozen HDV policy:    {args.hdv_model}")
        print(f"  HDV fallback action:  {args.hdv_action}")
    print(f"  initial ADV episodes: {args.initial_adv_episodes}")
    print(f"  initial EGO episodes: {args.initial_ego_episodes}")
    if args.ego_warmup:
        print("  ego warmup:           ENABLED")
        print(f"    episodes:           {args.ego_warmup_min_episodes}-{args.ego_warmup_max_episodes}")
        print(f"    rule adv action:    {args.warmup_adv_action}")
        print(f"    target crash:       <= {args.ego_warmup_target_crash_rate:.1%}")
        print(f"    target completion:  >= {args.ego_warmup_target_completion:.1%}")
        print(f"    target window:      {args.ego_warmup_window}")
    print(f"  eval_interval:        {args.eval_interval}")
    print(f"  save_interval:        {args.save_interval}")
    print(f"  output_dir:           {output_dir}")
    print("=" * 70)


def print_phase_header(round_no: int, phase_name: str, n_episodes: str, num_cav: int):
    print(f"\n{'#' * 70}")
    if phase_name == "ADVERSARIAL":
        print(f"  R{round_no} | PHASE: {phase_name} TRAINING")
        print(f"  {'[TRAINING]':>12} MAPPO  ({num_cav} adversarial vehicles)")
        print(f"  {'[FROZEN]':>12} EgoPPO (ego vehicle, inference only)")
    elif phase_name == "EGO WARMUP":
        print(f"  R{round_no} | PHASE: {phase_name}")
        print(f"  {'[RULE]':>12} CAVs   ({num_cav} rule-controlled vehicles)")
        print(f"  {'[TRAINING]':>12} EgoPPO (ego vehicle)")
    else:
        print(f"  R{round_no} | PHASE: {phase_name} TRAINING")
        print(f"  {'[FROZEN]':>12} MAPPO  ({num_cav} adversarial vehicles)")
        print(f"  {'[TRAINING]':>12} EgoPPO (ego vehicle)")
    print(f"  Episodes:              {n_episodes}")
    print(f"{'#' * 70}")


def print_round_summary(round_no: int, adv_info: dict, ego_info: dict):
    print(f"\n{'=' * 70}")
    print(f"  ROUND {round_no} SUMMARY")
    print(f"{'=' * 70}")
    print("  Adversarial Phase:")
    print(f"    Episodes:       {adv_info.get('episodes', 0)}")
    print(f"    Avg Reward:     {adv_info.get('avg_reward', 0):.2f}")
    print(f"    Ego Crash Rate: {adv_info.get('crash_rate', 0):.3f}")
    print(f"    Adv Crash Rate: {adv_info.get('agents_crash_rate', 0):.3f}")
    print("  Ego Phase:")
    print(f"    Episodes:       {ego_info.get('episodes', 0)}")
    print(f"    Avg Reward:     {ego_info.get('avg_reward', 0):.2f}")
    print(f"    Ego Crash Rate: {ego_info.get('crash_rate', 0):.3f}")
    print(f"    Adv Crash Rate: {ego_info.get('agents_crash_rate', 0):.3f}")
    print(f"    Road Compl. %:  {ego_info.get('road_completion_rate', 0):.1%}")
    print(f"    Avg Speed:      {ego_info.get('avg_speed', 0):.1f} m/s")
    print(f"{'=' * 70}")


def print_ego_update(losses):
    if not losses:
        return
    print(
        f"  [EGO] Update | "
        f"P_Loss: {losses.get('policy_loss', 0):.4f} | "
        f"V_Loss: {losses.get('value_loss', 0):.4f}"
    )


if __name__ == "__main__":
    main()
