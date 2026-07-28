"""Evaluate ego policies on fixed LC straight-follow scenes."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = MODEL_ROOT.parent
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from envs.action_adapter import ActionAdapter, DiscreteDrivingAction
from tests.envs.observation_adapters import OurEgo25DObservationAdapter
from tests.scenarios.parameter_space import StraightFollowParameterSpace
from tests.scenarios.reinforce_continuous import LCReinforcePolicy
from tests.scenarios.templates.base import apply_intent, detect_collision
from tests.scenarios.templates import TEMPLATE_REGISTRY, make_template


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ego policies on LC straight-road scenes.")
    parser.add_argument("--template-name", choices=sorted(TEMPLATE_REGISTRY.keys()), default="straight_follow")
    parser.add_argument("--lc-checkpoint", default=None, help="LC checkpoint used to generate scenes deterministically.")
    parser.add_argument("--scene-params-csv", default=None, help="Fixed decoded_scene_params.csv from LC training.")
    parser.add_argument("--num-scenes", type=int, default=20, help="Number of deterministic scenes to generate from LC checkpoint.")
    parser.add_argument("--ego-checkpoints", default="", help="Comma-separated EgoPPO checkpoints. Empty uses fixed ego action.")
    parser.add_argument("--ego-labels", default="", help="Comma-separated labels matching --ego-checkpoints.")
    parser.add_argument("--ego-action", choices=["keep_lane", "accelerate", "decelerate"], default="accelerate")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--route-length", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=669)
    parser.add_argument("--target-speed", type=float, default=18.0)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%b_%d_%H_%M_%S")
        output_dir = MODEL_ROOT / "tests" / "results" / f"lc_straight_eval_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ego_action_id(name: str) -> int:
    mapping = {
        "keep_lane": int(DiscreteDrivingAction.KEEP_LANE),
        "accelerate": int(DiscreteDrivingAction.ACCELERATE),
        "decelerate": int(DiscreteDrivingAction.DECELERATE),
    }
    return mapping[str(name)]


def parse_json_field(value: str, default):
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            import ast

            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return default


def load_fixed_scenes(path: str) -> List[Dict]:
    scenes = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            decoded_params = parse_json_field(row.get("decoded_params", ""), {})
            if not decoded_params:
                decoded_params = {
                    key.replace("param_", ""): value
                    for key, value in row.items()
                    if key.startswith("param_") and value not in {"", None}
                }
            for key, value in list(decoded_params.items()):
                if key in {"behavior_type", "template_name"}:
                    continue
                try:
                    decoded_params[key] = float(value)
                except (TypeError, ValueError):
                    pass
            lc_action = parse_json_field(row.get("lc_action", ""), [])
            scenes.append({
                "scene_id": int(row.get("episode", len(scenes) + 1)),
                "seed": int(float(row.get("seed", len(scenes)))),
                "template_name": row.get("template_name", "straight_follow") or "straight_follow",
                "behavior_type": decoded_params.get("behavior_type", row.get("behavior_type", "")),
                "lc_action": lc_action,
                "decoded_params": decoded_params,
            })
    return scenes


def generate_scenes_from_lc(args, template) -> List[Dict]:
    if not args.lc_checkpoint:
        raise ValueError("Either --scene-params-csv or --lc-checkpoint is required.")
    policy = LCReinforcePolicy({
        "num_scenario": 1,
        "batch_size": 1,
        "model_path": str(Path(args.lc_checkpoint).parent),
        "device": "cpu" if args.no_cuda else None,
    })
    policy.load_model(args.lc_checkpoint)
    policy.set_mode("eval")
    parameter_space = StraightFollowParameterSpace(ego_speed=args.target_speed)
    static_state = [template.static_lc_state(target_speed=args.target_speed)]
    scenes = []
    for index in range(int(args.num_scenes)):
        lc_action, _ = policy.get_init_action(static_state, deterministic=True)
        decoded = parameter_space.decode(lc_action[0], template_name=args.template_name)
        scenes.append({
            "scene_id": index + 1,
            "seed": int(args.seed + index),
            "template_name": args.template_name,
            "behavior_type": decoded["behavior_type"],
            "lc_action": list(np.asarray(lc_action[0], dtype=float)),
            "decoded_params": decoded,
        })
    return scenes


def checkpoint_step(path: Optional[Path]) -> int:
    if path is None:
        return -1
    match = re.search(r"checkpoint-(\d+)\.pt$", str(path))
    if match:
        return int(match.group(1))
    match = re.search(r"model\.sac\.(\d+)\.torch$", str(path))
    return int(match.group(1)) if match else -1


def resolve_ego_checkpoints(value: str, labels: str = "") -> List[Tuple[str, Optional[Path]]]:
    items = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not items:
        return [("fixed_action", None)]
    label_items = [item.strip() for item in str(labels or "").split(",") if item.strip()]
    result = []
    for index, item in enumerate(items):
        path = Path(item)
        label = label_items[index] if index < len(label_items) else path.stem
        result.append((label, path))
    return result


def load_ego_policy(path: Optional[Path], no_cuda: bool, seed: int):
    if path is None:
        return None
    if str(path).endswith(".zip"):
        from tests.envs.sb3_ego_adapter import SB3ContinuousEgoAsDiscretePolicy

        return SB3ContinuousEgoAsDiscretePolicy(path)
    from carla_evolution.agents.ego_ppo import EgoPPOAdapter

    ego = EgoPPOAdapter.from_config({"use_cuda": not bool(no_cuda), "seed": int(seed)}, state_dim=25, action_dim=5)
    ego.load(str(path))
    return ego


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


def run_episode(template, scene_spec: Dict, ego_policy, ego_action: int, args, action_adapter: ActionAdapter):
    decoded_params = dict(scene_spec["decoded_params"])
    scene = template.reset(decoded_params, seed=int(scene_spec["seed"]))
    obs_adapter = OurEgo25DObservationAdapter(max_lanes=template.max_lanes)
    episode_reward = 0.0
    stats = init_episode_stats()
    collision = False
    success = False
    termination_reason = "max_steps"

    for step in range(int(args.max_steps)):
        ego_obs = obs_adapter.build(scene)
        if ego_policy is None:
            action = ego_action
        else:
            action, _, _ = ego_policy.select_action(ego_obs, deterministic=True)
        ego_intent = action_adapter.to_control_intent(int(action))
        npc_intents = template.npc_intents(scene, step=step, dt=float(args.dt))

        apply_intent(scene.ego_vehicle, ego_intent, dt=float(args.dt))
        for vehicle, intent in zip(scene.adv_cav_vehicles, npc_intents):
            apply_intent(vehicle, intent, dt=float(args.dt))

        scene.sync_route_progress_from_ego()
        collision = detect_collision(scene.vehicles)
        scene.events.collision = bool(collision)
        success = bool(scene.route_completion >= 1.0)
        update_stats(stats, scene)

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
    }


def write_csv(path: Path, records: Iterable[Dict]) -> None:
    records = list(records)
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def summarize(records: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        grouped.setdefault(str(record["ego_model"]), []).append(record)
    rows = []
    for ego_model, items in grouped.items():
        rows.append({
            "ego_model": ego_model,
            "episodes": len(items),
            "mean_reward": float(np.mean([float(item["episode_reward"]) for item in items])),
            "collision_rate": float(np.mean([float(item["collision"]) for item in items])),
            "success_rate": float(np.mean([float(item["success"]) for item in items])),
            "mean_route_completion": float(np.mean([float(item["route_completion"]) for item in items])),
            "mean_average_speed": float(np.mean([float(item["average_speed"]) for item in items])),
            "mean_low_speed_ratio": float(np.mean([float(item["low_speed_ratio"]) for item in items])),
            "mean_progress_auc": float(np.mean([float(item["progress_auc"]) for item in items])),
        })
    return rows


def format_mean_var(mean_value, var_value) -> str:
    return f"{float(mean_value):.4f} ± {float(var_value):.4f}"


def scenario_family_name(template_name: str) -> str:
    return "follow_straight" if str(template_name) == "straight_follow" else str(template_name)


def scenario_family_summary(records: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, int, str], List[Dict]] = {}
    for record in records:
        key = (
            str(record["ego_model"]),
            int(record.get("checkpoint_step", -1)),
            str(record["scenario_family"]),
        )
        grouped.setdefault(key, []).append(record)
    rows = []
    for (model_label, step, family), items in grouped.items():
        row = {
            "model_group": "training_final",
            "model_label": model_label,
            "checkpoint_step": step,
            "scenario_family": family,
            "episodes": len(items),
        }
        metric_specs = [
            ("crash", "Collision rate"),
            ("route_completion", "Road completion"),
            ("ego_route_distance_m", "Driven distance"),
            ("mean_speed", "Average speed"),
            ("combined_efficiency", "Traffic efficiency"),
            ("episode_reward", "Average reward"),
        ]
        for key, label in metric_specs:
            values = np.asarray([float(item[key]) for item in items], dtype=np.float64)
            row[f"{label}_mean_var"] = format_mean_var(float(values.mean()), float(values.var()))
        rows.append(row)
    order = {"follow_straight": 0, "passing": 1, "lane_change": 2, "cut_in": 3}
    return sorted(rows, key=lambda item: (item["checkpoint_step"], order.get(item["scenario_family"], 99)))


def main():
    args = parse_args()
    output_dir = make_output_dir(args)
    template = make_template(args.template_name, route_length=args.route_length)
    scenes = load_fixed_scenes(args.scene_params_csv) if args.scene_params_csv else generate_scenes_from_lc(args, template)
    ego_items = resolve_ego_checkpoints(args.ego_checkpoints, args.ego_labels)
    action_adapter = ActionAdapter({})
    fixed_ego_action = ego_action_id(args.ego_action)

    records = []
    print("=" * 72)
    print("LC straight-road evaluation")
    print(f"  scenes:        {len(scenes)}")
    print(f"  ego_models:    {len(ego_items)}")
    print(f"  output_dir:    {output_dir}")
    print("=" * 72)

    for ego_label, ego_path in ego_items:
        ego_policy = load_ego_policy(ego_path, no_cuda=args.no_cuda, seed=args.seed)
        for scene_index, scene_spec in enumerate(scenes):
            scene_template = make_template(scene_spec["template_name"], route_length=args.route_length)
            metrics = run_episode(scene_template, scene_spec, ego_policy, fixed_ego_action, args, action_adapter)
            row = {
                "ego_model": ego_label,
                "model_label": ego_label,
                "model_group": "training_final",
                "checkpoint_step": checkpoint_step(ego_path),
                "ego_checkpoint": "" if ego_path is None else str(ego_path),
                "scene_id": scene_spec["scene_id"],
                "seed": scene_spec["seed"],
                "template_name": scene_spec["template_name"],
                "scenario_family": scenario_family_name(scene_spec["template_name"]),
                "behavior_type": scene_spec["behavior_type"],
                "lc_action": json.dumps(scene_spec.get("lc_action", [])),
                "decoded_params": json.dumps(scene_spec.get("decoded_params", {}), ensure_ascii=False, sort_keys=True),
                "crash": bool(metrics["collision"]),
                "mean_speed": float(metrics["average_speed"]),
                "ego_route_distance_m": float(metrics["driven_distance"]),
                "combined_efficiency": float(0.8 * metrics["progress_auc"] + 0.2 * (1.0 - metrics["low_speed_ratio"])),
                **metrics,
            }
            records.append(row)
            print(
                f"ego={ego_label} scene={scene_index + 1:04d} "
                f"behavior={row['behavior_type']} reward={row['episode_reward']:.2f} "
                f"collision={row['collision']} route={row['route_completion']:.3f}",
                flush=True,
            )

    episode_log = output_dir / "lc_eval_episode_log.csv"
    summary_log = output_dir / "lc_eval_model_summary.csv"
    scenario_family_log = output_dir / "scenario_family_comparison_mean_plus_variance.csv"
    write_csv(episode_log, records)
    write_csv(summary_log, summarize(records))
    write_csv(scenario_family_log, scenario_family_summary(records))
    print(f"Episode log: {episode_log}")
    print(f"Model summary: {summary_log}")
    print(f"Scenario family table: {scenario_family_log}")


if __name__ == "__main__":
    main()
