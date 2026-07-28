"""Train LC scenario policy on straight-follow test templates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = MODEL_ROOT.parent
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from envs.action_adapter import ActionAdapter, DiscreteDrivingAction
from tests.envs.replay_buffer import LCReplayBuffer
from tests.envs.observation_adapters import OurEgo25DObservationAdapter
from tests.scenarios.parameter_space import StraightFollowParameterSpace
from tests.scenarios.reinforce_continuous import LCReinforcePolicy
from tests.scenarios.templates.base import apply_intent, detect_collision
from tests.scenarios.templates import TEMPLATE_REGISTRY, make_template


def parse_args():
    parser = argparse.ArgumentParser(description="Train LC policy for straight-road templates.")
    parser.add_argument("--template-name", choices=sorted(TEMPLATE_REGISTRY.keys()), default="straight_follow")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--route-length", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=669)
    parser.add_argument("--target-speed", type=float, default=18.0)
    parser.add_argument("--ego-action", choices=["keep_lane", "accelerate", "decelerate"], default="accelerate")
    parser.add_argument("--ego-checkpoint", default="", help="Optional EgoPPO/SB3 checkpoint used during LC training.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-weight", type=float, default=1e-4)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-interval", type=int, default=50)
    return parser.parse_args()


def ego_action_id(name: str) -> int:
    mapping = {
        "keep_lane": int(DiscreteDrivingAction.KEEP_LANE),
        "accelerate": int(DiscreteDrivingAction.ACCELERATE),
        "decelerate": int(DiscreteDrivingAction.DECELERATE),
    }
    return mapping[str(name)]


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%b_%d_%H_%M_%S")
        output_dir = MODEL_ROOT / "tests" / "results" / f"lc_straight_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    return output_dir


def init_episode_stats() -> Dict[str, float]:
    return {
        "speed_sum": 0.0,
        "speed_min": float("inf"),
        "speed_max": 0.0,
        "low_speed_steps": 0,
        "progress_auc_sum": 0.0,
    }


def update_stats(stats: Dict[str, float], scene, low_speed_threshold: float = 2.0) -> None:
    ego = scene.ego_vehicle
    speed = float(ego.speed if ego is not None else 0.0)
    stats["speed_sum"] += speed
    stats["speed_min"] = min(stats["speed_min"], speed)
    stats["speed_max"] = max(stats["speed_max"], speed)
    stats["low_speed_steps"] += int(speed < low_speed_threshold)
    stats["progress_auc_sum"] += float(scene.route_completion)


def load_ego_policy(path: str, no_cuda: bool, seed: int):
    if not str(path or "").strip():
        return None
    if str(path).endswith(".zip"):
        from tests.envs.sb3_ego_adapter import SB3ContinuousEgoAsDiscretePolicy

        return SB3ContinuousEgoAsDiscretePolicy(path)
    from carla_evolution.agents.ego_ppo import EgoPPOAdapter

    ego = EgoPPOAdapter.from_config({"use_cuda": not bool(no_cuda), "seed": int(seed)}, state_dim=25, action_dim=5)
    ego.load(str(path))
    return ego


def run_episode(template, decoded_params, lc_action, args, seed: int, action_adapter: ActionAdapter, ego_policy=None):
    scene = template.reset(decoded_params, seed=seed)
    obs_adapter = OurEgo25DObservationAdapter(max_lanes=template.max_lanes)
    ego_action = ego_action_id(args.ego_action)
    episode_reward = 0.0
    stats = init_episode_stats()
    collision = False
    success = False
    termination_reason = "max_steps"

    for step in range(int(args.max_steps)):
        ego_obs = obs_adapter.build(scene)
        if ego_policy is None:
            action_id = ego_action
        else:
            action_id, _, _ = ego_policy.select_action(ego_obs, deterministic=True)
        ego_intent = action_adapter.to_control_intent(int(action_id))
        npc_intents = template.npc_intents(scene, step=step, dt=float(args.dt))

        apply_intent(scene.ego_vehicle, ego_intent, dt=float(args.dt))
        for vehicle, intent in zip(scene.adv_cav_vehicles, npc_intents):
            apply_intent(vehicle, intent, dt=float(args.dt))

        scene.sync_route_progress_from_ego()
        collision = detect_collision(scene.vehicles)
        scene.events.collision = bool(collision)
        success = bool(scene.route_completion >= 1.0)
        update_stats(stats, scene)

        # LC keeps SafeBench's objective through a negative episode reward;
        # this ego reward is a lightweight test reward for the straight rollout.
        step_reward = float(scene.ego_vehicle.speed) - (50.0 if collision else 0.0)
        episode_reward += step_reward

        if collision:
            termination_reason = "collision"
            break
        if success:
            termination_reason = "success"
            break

    steps = step + 1
    return {
        "episode_reward": float(episode_reward),
        "collision": bool(collision),
        "success": bool(success),
        "timeout": bool(not collision and not success and steps >= int(args.max_steps)),
        "route_completion": float(scene.route_completion),
        "average_speed": float(stats["speed_sum"] / max(steps, 1)),
        "min_speed": float(0.0 if stats["speed_min"] == float("inf") else stats["speed_min"]),
        "max_speed": float(stats["speed_max"]),
        "low_speed_ratio": float(stats["low_speed_steps"] / max(steps, 1)),
        "progress_auc": float(stats["progress_auc_sum"] / max(steps, 1)),
        "driven_distance": float(scene.route.ego_progress),
        "episode_steps": int(steps),
        "termination_reason": termination_reason,
        "lc_action": list(np.asarray(lc_action, dtype=float)),
        "decoded_params": dict(decoded_params),
    }


def append_csv(path: Path, row: Dict) -> None:
    write_header = not path.exists()
    fieldnames = sorted(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


SCENE_PARAM_FIELDS = [
    "episode",
    "seed",
    "template_name",
    "behavior_type",
    "lc_action",
    "decoded_params",
    "front_offset",
    "front_initial_speed",
    "trigger_distance",
    "decel_target_speed",
    "oscillation_base_speed",
    "oscillation_amplitude",
    "oscillation_period",
    "hard_brake_strength",
    "hard_brake_duration",
    "recovery_speed",
    "slow_front_offset",
    "slow_front_speed",
    "target_lane",
    "target_lane_front_offset",
    "target_lane_front_speed",
    "passing_gap",
    "blocker_offset",
    "blocker_speed",
    "target_front_offset",
    "target_front_speed",
    "target_rear_offset",
    "target_rear_speed",
    "target_lane_gap",
    "source_lane",
    "cut_in_offset",
    "cut_in_speed",
    "cut_in_target_speed",
    "lead_offset",
    "lead_speed",
    "lead_gap",
]


def append_scene_csv(path: Path, row: Dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCENE_PARAM_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SCENE_PARAM_FIELDS})


def main():
    args = parse_args()
    output_dir = make_output_dir(args)
    train_log = output_dir / "lc_train_log.csv"
    scene_log = output_dir / "decoded_scene_params.csv"

    template = make_template(args.template_name, route_length=args.route_length)
    parameter_space = StraightFollowParameterSpace(ego_speed=args.target_speed)
    action_adapter = ActionAdapter({})
    device = "cpu" if args.no_cuda else None
    policy = LCReinforcePolicy({
        "num_scenario": 1,
        "batch_size": args.batch_size,
        "buffer_capacity": args.buffer_capacity,
        "lr": args.lr,
        "entropy_weight": args.entropy_weight,
        "model_path": str(output_dir / "models"),
        "device": device,
    })
    buffer = LCReplayBuffer(num_scenario=1, buffer_capacity=args.buffer_capacity)
    static_state = [template.static_lc_state(target_speed=args.target_speed)]
    ego_policy = load_ego_policy(args.ego_checkpoint, no_cuda=args.no_cuda, seed=args.seed)

    print("=" * 72)
    print(f"LC {args.template_name} training")
    print(f"  episodes:      {args.episodes}")
    print(f"  max_steps:     {args.max_steps}")
    print(f"  template:      {args.template_name}")
    print(f"  ego_checkpoint:{args.ego_checkpoint or '(fixed ' + args.ego_action + ')'}")
    print(f"  output_dir:    {output_dir}")
    print("=" * 72)

    for episode in range(int(args.episodes)):
        seed = int(args.seed + episode)
        lc_action, additional = policy.get_init_action(static_state, deterministic=False)
        decoded = parameter_space.decode(lc_action[0], template_name=args.template_name)
        metrics = run_episode(template, decoded, lc_action[0], args, seed, action_adapter, ego_policy=ego_policy)

        processed_static = policy.process_init_state(static_state)
        buffer.store_init([processed_static, lc_action], additional)
        buffer.add_episode_reward(metrics["episode_reward"])
        train_result = policy.train(buffer) or {}

        row = {
            "episode": episode + 1,
            "seed": seed,
            "template_name": args.template_name,
            "behavior_type": decoded["behavior_type"],
            "episode_reward": metrics["episode_reward"],
            "lc_reward": -metrics["episode_reward"],
            "collision": metrics["collision"],
            "route_completion": metrics["route_completion"],
            "average_speed": metrics["average_speed"],
            "min_speed": metrics["min_speed"],
            "max_speed": metrics["max_speed"],
            "low_speed_ratio": metrics["low_speed_ratio"],
            "progress_auc": metrics["progress_auc"],
            "episode_steps": metrics["episode_steps"],
            "termination_reason": metrics["termination_reason"],
            "loss": train_result.get("loss", ""),
        }
        append_csv(train_log, row)

        scene_row = {
            "episode": episode + 1,
            "seed": seed,
            "template_name": args.template_name,
            "behavior_type": decoded["behavior_type"],
            "lc_action": json.dumps(metrics["lc_action"]),
            "decoded_params": json.dumps(metrics["decoded_params"], ensure_ascii=False, sort_keys=True),
        }
        scene_row.update(decoded)
        append_scene_csv(scene_log, scene_row)

        if args.save_interval > 0 and (episode + 1) % int(args.save_interval) == 0:
            policy.save_model(epoch=episode + 1)

        if (episode + 1) % max(1, min(10, int(args.episodes))) == 0:
            print(
                f"episode {episode + 1:04d} | "
                f"behavior={decoded['behavior_type']} "
                f"reward={metrics['episode_reward']:.2f} "
                f"lc_reward={-metrics['episode_reward']:.2f} "
                f"collision={metrics['collision']} "
                f"loss={train_result.get('loss', '')}"
            )

    final_path = policy.save_model(epoch=int(args.episodes))
    print(f"Saved final LC model: {final_path}")
    print(f"Train log: {train_log}")
    print(f"Scene params: {scene_log}")


if __name__ == "__main__":
    main()
