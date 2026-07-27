"""Evaluate selected EgoPPO and MAPPO checkpoints without HDV traffic."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = MODEL_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))


SCENARIO_PRESETS: Dict[str, Dict] = {
    "follow_straight": {
        "route_name": "lc_follow_straight",
        "scenario_family": "follow_straight",
        "behavior_type": "mixed_follow",
        "route_length": 200.0,
        "ego_offset": 0.0,
        "ego_lane_offset": 1,
        "adv_offsets": [30.0, 55.0, -35.0],
        "adv_lane_offsets": [1, 1, 0],
        "target_speed_cruise": 20.0,
        "relative_offset_jitter": 0.0,
        "randomize_spawn_start_index": False,
        "shuffle_spawn_points": False,
    },
    "passing": {
        "route_name": "lc_passing",
        "scenario_family": "passing",
        "behavior_type": "slow_front_overtake",
        "route_length": 200.0,
        "ego_offset": 0.0,
        "ego_lane_offset": 1,
        "adv_offsets": [32.0, 58.0, -38.0],
        "adv_lane_offsets": [1, 0, 2],
        "target_speed_cruise": 20.0,
        "relative_offset_jitter": 0.0,
        "randomize_spawn_start_index": False,
        "shuffle_spawn_points": False,
    },
    "lane_change": {
        "route_name": "lc_lane_change",
        "scenario_family": "lane_change",
        "behavior_type": "right_escape",
        "route_length": 200.0,
        "ego_offset": 0.0,
        "ego_lane_offset": 1,
        "adv_offsets": [28.0, 8.0, -40.0],
        "adv_lane_offsets": [1, 2, 0],
        "target_speed_cruise": 20.0,
        "relative_offset_jitter": 0.0,
        "randomize_spawn_start_index": False,
        "shuffle_spawn_points": False,
    },
    "cut_in": {
        "route_name": "lc_cut_in",
        "scenario_family": "cut_in",
        "behavior_type": "cut_in_recovery",
        "route_length": 200.0,
        "ego_offset": 0.0,
        "ego_lane_offset": 1,
        "adv_offsets": [18.0, 55.0, -35.0],
        "adv_lane_offsets": [2, 1, 0],
        "target_speed_cruise": 20.0,
        "relative_offset_jitter": 0.0,
        "randomize_spawn_start_index": False,
        "shuffle_spawn_points": False,
    },
    "slow_front_left_escape": {
        "route_name": "conservative_slow_front_left_escape",
        "scenario_family": "passing",
        "behavior_type": "slow_front_overtake",
        "route_length": 200.0,
        "ego_offset": 0.0,
        "ego_lane_offset": 1,
        "adv_offsets": [32.0, -38.0, 15.0],
        "adv_lane_offsets": [1, 2, 0],
        "target_speed_cruise": 20.0,
        "relative_offset_jitter": 0.0,
        "randomize_spawn_start_index": False,
        "shuffle_spawn_points": False,
    },
    "cut_in_recovery": {
        "route_name": "conservative_cut_in_recovery",
        "scenario_family": "cut_in",
        "behavior_type": "cut_in_recovery",
        "route_length": 200.0,
        "ego_offset": 0.0,
        "ego_lane_offset": 1,
        "adv_offsets": [18.0, 55.0, -35.0],
        "adv_lane_offsets": [2, 1, 0],
        "target_speed_cruise": 20.0,
        "relative_offset_jitter": 0.0,
        "randomize_spawn_start_index": False,
        "shuffle_spawn_points": False,
    },
    "partial_surround_right_escape": {
        "route_name": "conservative_partial_surround_right_escape",
        "scenario_family": "lane_change",
        "behavior_type": "right_escape",
        "route_length": 200.0,
        "ego_offset": 0.0,
        "ego_lane_offset": 1,
        "adv_offsets": [28.0, 8.0, -40.0],
        "adv_lane_offsets": [1, 2, 0],
        "target_speed_cruise": 20.0,
        "relative_offset_jitter": 0.0,
        "randomize_spawn_start_index": False,
        "shuffle_spawn_points": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a selected ego checkpoint against a selected adversarial MAPPO checkpoint."
    )
    parser.add_argument("--backend", choices=["mock", "carla"], default="carla")
    parser.add_argument("--config", default=str(MODEL_ROOT / "configs" / "carla_0915.yaml"))
    parser.add_argument("--adv-model-dir", required=True, help="Directory containing MAPPO actor_*.pt and critic_*.pt.")
    parser.add_argument("--adv-step", type=int, default=None, help="MAPPO checkpoint step. Defaults to latest common step.")
    parser.add_argument(
        "--ego-checkpoint",
        required=True,
        help="EgoPPO checkpoint path, or a directory containing checkpoint-N.pt files.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--num-cav", type=int, default=3)
    parser.add_argument(
        "--scenario-preset",
        choices=sorted(SCENARIO_PRESETS),
        default=None,
        help="Fixed initialization preset for conservative ego testing.",
    )
    parser.add_argument("--spawn-start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=669)
    parser.add_argument("--scenario-seed-span", type=int, default=1000000)
    parser.add_argument("--no-scenario-randomization", action="store_true", default=False)
    parser.add_argument("--torch-seed", type=int, default=669)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--carla-rpc-timeout", type=float, default=180.0)
    parser.add_argument("--render", action="store_true", default=False, help="Enable CARLA rendering and spectator follow.")
    parser.add_argument("--run-label", default="", help="Optional label copied into per-episode CSV rows.")
    parser.add_argument("--output-csv", default=None, help="Optional per-episode CSV output path.")
    return parser.parse_args()


def episode_seed(args, episode_index: int) -> int:
    if args.no_scenario_randomization:
        return int(args.seed + episode_index)
    span = max(int(args.scenario_seed_span), 1)
    rng = np.random.RandomState(int(args.seed) + int(episode_index) * 9973)
    return int(rng.randint(0, span))


def resolve_checkpoints(args):
    from carla_evolution.training import train as train_mod

    adv_model_dir = Path(args.adv_model_dir)
    if not adv_model_dir.exists():
        raise FileNotFoundError(f"MAPPO model directory not found: {adv_model_dir}")

    adv_step = args.adv_step
    if adv_step is None:
        adv_step = train_mod.latest_mappo_step(str(adv_model_dir))

    ego_checkpoint = Path(args.ego_checkpoint)
    if ego_checkpoint.is_dir():
        ego_checkpoint = Path(train_mod.latest_ego_checkpoint(str(ego_checkpoint)))
    if not ego_checkpoint.exists():
        raise FileNotFoundError(f"Ego checkpoint not found: {ego_checkpoint}")

    return adv_model_dir, int(adv_step), ego_checkpoint


def build_env(args):
    from carla_evolution.envs.factory import make_env

    env_config = {
        "backend": args.backend,
        "max_episode_steps": int(args.max_steps),
        "num_cav": int(args.num_cav),
        "num_hdv": 0,
        "num_background": 0,
        "timeout": float(args.carla_rpc_timeout),
        "randomize_scenarios": not bool(args.no_scenario_randomization),
        "spawn_start_index": int(args.spawn_start_index),
    }
    if args.render:
        env_config.update({
            "no_rendering_mode": False,
            "spectator_follow": True,
        })
    if args.scenario_preset:
        env_config.update(SCENARIO_PRESETS[args.scenario_preset])
    return make_env(env_config, config_path=args.config)


def load_policies(args, env, adv_model_dir: Path, adv_step: int, ego_checkpoint: Path):
    from carla_evolution.agents.ego_ppo import EgoPPOAdapter
    from carla_evolution.agents.mappo import MAPPOAgent

    mappo = MAPPOAgent.from_config(
        {
            "reward_type": "agents_rewards",
            "use_cuda": not bool(args.no_cuda),
            "torch_seed": int(args.torch_seed),
        },
        state_dim=env.n_s,
        action_dim=env.n_a,
    )
    ego = EgoPPOAdapter.from_config(
        {"use_cuda": not bool(args.no_cuda), "seed": int(args.torch_seed)},
        state_dim=env.n_s,
        action_dim=env.n_a,
    )
    mappo.load(str(adv_model_dir), int(adv_step), train_mode=False)
    ego.load(str(ego_checkpoint))
    return mappo, ego


def run_eval_episode(env, mappo, ego, args, seed: int):
    from carla_evolution.training import train as train_mod

    state, reset_info = env.reset(seed=seed)
    n_agents = len(env.controlled_vehicles)
    if n_agents <= 0:
        raise RuntimeError("Evaluation requires at least one adversarial CAV.")

    ego_state = np.asarray(env.obs2, dtype=np.float32).flatten()
    episode_reward = 0.0
    last_info = {"route_completion": reset_info.get("route_completion", 0.0)}
    diagnostics = train_mod.init_episode_diagnostics()

    for step in range(int(args.max_steps)):
        adv_actions = mappo.action(state, n_agents)
        ego_action, _, _ = ego.select_action(ego_state, deterministic=True)
        full_action = list(adv_actions) + [ego_action]
        next_state, global_reward, terminated, truncated, info = env.step(full_action)
        next_ego_state = np.asarray(env.obs2, dtype=np.float32).flatten()

        ego_reward = float(info.get("ego_reward", global_reward))
        episode_reward += ego_reward
        last_info = info
        train_mod.update_episode_diagnostics(diagnostics, info)

        state = next_state
        ego_state = next_ego_state
        if terminated or truncated:
            break

    reward_array = np.asarray(last_info.get("agents_rewards", [0.0]), dtype=np.float32)
    metrics = train_mod.episode_metrics(
        step + 1,
        episode_reward,
        reward_array,
        train_mod.merge_episode_diagnostics(last_info, diagnostics),
    )
    metrics["seed"] = int(seed)
    metrics["num_cav"] = int(n_agents)
    metrics["scenario_preset"] = str(args.scenario_preset or "")
    metrics["run_label"] = str(args.run_label or "")
    return metrics


def summarize(records):
    if not records:
        return {}
    rewards = [float(item.get("episode_reward", 0.0)) for item in records]
    return {
        "episodes": len(records),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "crash_rate": float(np.mean([float(item.get("crash", False)) for item in records])),
        "adv_collision_rate": float(np.mean([float(item.get("agents_crash", False)) for item in records])),
        "road_completion_rate": float(np.mean([float(item.get("route_completion", 0.0)) for item in records])),
        "road_completion_rate_full": float(
            np.mean([float(item.get("route_completion_full", item.get("route_completion", 0.0))) for item in records])
        ),
        "route_completion_distance": float(np.mean([float(item.get("route_completion_distance", 0.0)) for item in records])),
        "route_length": float(np.mean([float(item.get("route_length", 0.0)) for item in records])),
        "avg_speed": float(np.mean([float(item.get("average_speed", 0.0)) for item in records])),
        "mean_length": float(np.mean([float(item.get("steps", 0)) for item in records])),
    }


def write_csv(path: str, records):
    if not path or not records:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record.keys()})
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_args()
    adv_model_dir, adv_step, ego_checkpoint = resolve_checkpoints(args)
    env = build_env(args)
    try:
        mappo, ego = load_policies(args, env, adv_model_dir, adv_step, ego_checkpoint)
        records = []
        for episode in range(int(args.episodes)):
            seed = episode_seed(args, episode)
            metrics = run_eval_episode(env, mappo, ego, args, seed)
            records.append(metrics)
            print(
                "episode={:04d} seed={} reward={:.3f} crash={} route={:.3f} avg_speed={:.3f}".format(
                    episode + 1,
                    seed,
                    float(metrics.get("episode_reward", 0.0)),
                    bool(metrics.get("crash", False)),
                    float(metrics.get("route_completion", 0.0)),
                    float(metrics.get("average_speed", 0.0)),
                ),
                flush=True,
            )
    finally:
        env.close()

    write_csv(args.output_csv, records)
    summary = summarize(records)
    print("summary", summary, flush=True)


if __name__ == "__main__":
    main()
