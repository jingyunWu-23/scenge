#!/usr/bin/env python3
"""Evaluate ego checkpoints with SafeBench-style scenario_id/route/init cases."""

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
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = MODEL_ROOT.parent
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from envs.action_adapter import ActionAdapter, DiscreteDrivingAction
from tests.envs.observation_adapters import OurEgo25DObservationAdapter
from tests.scenarios.templates.base import apply_intent, detect_collision
from tests.scenarios.templates import make_template


DEFAULT_TRAINING_EGO = MODEL_ROOT / "results" / "joint_Jul_01_14_59_59" / "models" / "ego" / "checkpoint-62710.pt"
DEFAULT_FINETUNE_DIR = (
    MODEL_ROOT
    / "results"
    / "ego_enhanced_poet_finetune"
    / "poet_finetune_Jul_21_13_29_28-30-10"
    / "models"
    / "ego"
)

DEFAULT_SAFEBENCH_ROOT = Path("/home/chenyuanwan/download/SafeBench-main/safebench")
DEFAULT_SCENARIO_TYPE_JSON = DEFAULT_SAFEBENCH_ROOT / "scenario" / "config" / "scenario_type" / "adv_init_state.json"

SAFEBENCH_SCENARIO_NAMES = {
    1: "DynamicObjectCrossing",
    2: "VehicleTurningRoute",
    3: "OtherLeadingVehicle",
    4: "ManeuverOppositeDirection",
    5: "OppositeVehicleRunningRedLight",
    6: "SignalizedJunctionLeftTurn",
    7: "SignalizedJunctionRightTurn",
    8: "NoSignalJunctionCrossingRoute",
}

FOLLOW_BEHAVIORS = ["sudden_decel", "speed_oscillation", "hard_brake"]

METRICS = [
    ("route_completion", "Road completion"),
    ("crash", "Ego crash rate"),
    ("combined_efficiency", "Traffic efficiency"),
    ("ego_reward", "Ego reward"),
    ("episode_reward", "Average reward"),
    ("mean_speed", "Average speed"),
    ("ego_route_distance_m", "Driven distance"),
]

BAR_METRICS = [
    ("combined_efficiency", "Traffic efficiency"),
    ("ego_route_distance_m", "Driven distance"),
    ("mean_speed", "Average speed"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ego-checkpoint", default=str(DEFAULT_TRAINING_EGO))
    parser.add_argument("--training-ego-label", default="", help="Defaults to Ego-{checkpoint step}-train.")
    parser.add_argument("--finetune-ego-dir", default=str(DEFAULT_FINETUNE_DIR))
    parser.add_argument(
        "--finetune-ego-checkpoints",
        default="",
        help="Comma-separated fine-tuned EgoPPO checkpoints. Empty selects interval checkpoints from --finetune-ego-dir.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--include-final", action="store_true", default=True)
    parser.add_argument("--no-include-final", action="store_false", dest="include_final")
    parser.add_argument(
        "--scenario-type-json",
        default=str(DEFAULT_SCENARIO_TYPE_JSON),
        help="SafeBench scenario_type json. Defaults to LC.yaml's adv_init_state.json.",
    )
    parser.add_argument(
        "--scenario-ids",
        default="2",
        help="Comma-separated SafeBench scenario ids. Default 2 matches safebench/scenario/config/LC.yaml.",
    )
    parser.add_argument("--route-ids", default="", help="Comma-separated route ids. Empty uses all routes in the scenario_type json.")
    parser.add_argument("--max-routes-per-scenario", type=int, default=0, help="0 keeps all matched routes.")
    parser.add_argument("--max-inits-per-route", type=int, default=10, help="Maximum SafeBench parameter/init cases per route. 0 keeps all.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--route-length", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=669)
    parser.add_argument("--target-speed", type=float, default=18.0)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--summarize-only", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False, help="Show a lightweight 2D live view of selected episodes.")
    parser.add_argument("--render-delay", type=float, default=0.03, help="Seconds to pause between rendered steps.")
    parser.add_argument(
        "--render-first-n-per-model",
        type=int,
        default=1,
        help="When --render is enabled, render only the first N cases for each ego model. <=0 renders every case.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)\.pt$", str(path))
    return int(match.group(1)) if match else -1


def label_for_checkpoint(path: Path, suffix: str = "") -> str:
    step = checkpoint_step(path)
    label = f"Ego-{step}" if step > 0 else path.stem
    return f"{label}{suffix}" if suffix else label


def select_interval_checkpoints(model_dir: Path, interval: int, include_final: bool) -> List[Path]:
    checkpoints = sorted(model_dir.glob("checkpoint-*.pt"), key=checkpoint_step)
    checkpoints = [path for path in checkpoints if checkpoint_step(path) > 0]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-*.pt found under {model_dir}")

    selected = []
    last_step: Optional[int] = None
    for path in checkpoints:
        step = checkpoint_step(path)
        if last_step is None or step - last_step >= int(interval):
            selected.append(path)
            last_step = step
    if include_final and selected[-1] != checkpoints[-1]:
        selected.append(checkpoints[-1])
    return selected


def resolve_ego_checkpoints(args) -> List[Tuple[str, Path, str]]:
    training_path = Path(args.training_ego_checkpoint)
    if not training_path.exists():
        raise FileNotFoundError(str(training_path))
    training_label = args.training_ego_label.strip() or label_for_checkpoint(training_path, "-train")
    result = [(training_label, training_path, "training_final")]

    if args.finetune_ego_checkpoints.strip().lower() in {"none", "null", "skip"}:
        fine_paths = []
    elif args.finetune_ego_checkpoints.strip():
        fine_paths = [Path(item.strip()) for item in args.finetune_ego_checkpoints.split(",") if item.strip()]
    else:
        fine_paths = select_interval_checkpoints(Path(args.finetune_ego_dir), args.checkpoint_interval, args.include_final)
    for path in fine_paths:
        if not path.exists():
            raise FileNotFoundError(str(path))
        result.append((label_for_checkpoint(path), path, "finetune_sweep"))
    return result


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%b_%d_%H_%M_%S")
        output_dir = MODEL_ROOT / "tests" / "results" / f"safebench_lc_ego_sweep_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    return output_dir


def fixed_ego_action() -> int:
    return int(DiscreteDrivingAction.ACCELERATE)


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


class LiveTopDownRenderer:
    def __init__(self, enabled: bool, delay: float):
        self.enabled = bool(enabled)
        self.delay = float(delay)
        self.plt = None
        self.fig = None
        self.ax = None

    def close(self) -> None:
        if self.plt is not None and self.fig is not None:
            self.plt.close(self.fig)
        self.plt = None
        self.fig = None
        self.ax = None

    def draw(self, scene, title: str, step: int) -> None:
        if not self.enabled:
            return
        if self.plt is None:
            import matplotlib.pyplot as plt

            plt.ion()
            self.plt = plt
            self.fig, self.ax = plt.subplots(figsize=(10, 4))
        ax = self.ax
        ax.clear()
        route_length = float(scene.route.length)
        lane_width = float(scene.lane_width)
        max_lanes = int(scene.max_lanes)
        for lane_id in range(max_lanes):
            y = lane_id * lane_width
            ax.hlines(y, -5.0, route_length + 10.0, colors="#c9cdd4", linewidth=1.0, linestyles="--")
        for vehicle in scene.vehicles:
            color = "#f58518" if vehicle.role == "ego" else "#4c78a8"
            marker = "s" if vehicle.role == "ego" else "o"
            label = "ego" if vehicle.role == "ego" else f"npc-{vehicle.index}"
            ax.scatter([vehicle.route_s], [vehicle.lane_id * lane_width], s=90, c=color, marker=marker, label=label)
            ax.text(vehicle.route_s, vehicle.lane_id * lane_width + 0.35, f"{label} {vehicle.speed:.1f}", fontsize=8, ha="center")
        ego_s = float(scene.ego_vehicle.route_s if scene.ego_vehicle is not None else 0.0)
        ax.set_xlim(max(-5.0, ego_s - 35.0), min(route_length + 10.0, ego_s + 85.0))
        ax.set_ylim(-lane_width, lane_width * max_lanes)
        ax.set_xlabel("Route s (m)")
        ax.set_ylabel("Lane")
        ax.set_title(f"{title} | step={step} | completion={scene.route_completion:.3f}")
        ax.legend(loc="upper right", fontsize=8)
        self.fig.tight_layout()
        self.plt.pause(max(self.delay, 0.001))


def run_episode(template, scene_spec: Dict, ego_policy, ego_action: int, args, action_adapter: ActionAdapter):
    decoded_params = dict(scene_spec["decoded_params"])
    scene = template.reset(decoded_params, seed=int(scene_spec["seed"]))
    obs_adapter = OurEgo25DObservationAdapter(max_lanes=template.max_lanes)
    episode_reward = 0.0
    stats = init_episode_stats()
    collision = False
    success = False
    termination_reason = "max_steps"
    render_title = (
        f"{scene_spec.get('model_label', '')} "
        f"s{scene_spec.get('scenario_id', '')}/r{scene_spec.get('route_id', '')}/i{scene_spec.get('init_id', '')}"
    )
    renderer = LiveTopDownRenderer(bool(getattr(args, "_render_current_episode", False)), float(args.render_delay))

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
        renderer.draw(scene, render_title, step + 1)

        step_reward = float(scene.ego_vehicle.speed) - (50.0 if collision else 0.0)
        episode_reward += step_reward
        if collision:
            termination_reason = "collision"
            break
        if success:
            termination_reason = "success"
            break
    renderer.close()

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


def jitter(rng: np.random.RandomState, base: float, spread: float, low: float, high: float) -> float:
    return float(np.clip(base + rng.uniform(-spread, spread), low, high))


def parse_int_set(value: str) -> Optional[set]:
    items = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not items:
        return None
    return {int(item) for item in items}


def parameter_label(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def parameter_index(value, fallback: int) -> int:
    label = parameter_label(value)
    matches = re.findall(r"(\d+)", label)
    return int(matches[-1]) if matches else int(fallback)


def synthetic_params_from_safebench_case(scenario_id: int, route_id: int, init_id: int, args) -> Tuple[str, Dict]:
    """Map a SafeBench case id to the local straight-road template parameters.

    This preserves SafeBench's benchmark indexing. The current migrated test
    surface is still the local straight-road LC template, so CARLA route geometry
    is represented by deterministic route-dependent parameter shifts.
    """
    route_bucket = int(route_id) % 4
    init_bucket = int(init_id) % 10
    rng = np.random.RandomState(int(args.seed) + int(scenario_id) * 100000 + int(route_id) * 1000 + init_bucket)
    ego_speed = float(args.target_speed)

    if route_bucket == 3:
        slow_front_offset = jitter(rng, 26.0 + 1.5 * route_bucket, 4.0, 18.0, 45.0)
        target_lane_front_offset = slow_front_offset + jitter(rng, 22.0 + 0.8 * init_bucket, 5.0, 15.0, 45.0)
        params = {
            "template_name": "passing",
            "behavior_type": "passing",
            "ego_lane": 1,
            "target_lane": 0,
            "ego_speed": ego_speed,
            "slow_front_offset": slow_front_offset,
            "slow_front_speed": jitter(rng, 5.0 + 0.25 * init_bucket, 1.0, 3.0, 10.0),
            "target_lane_front_offset": target_lane_front_offset,
            "target_lane_front_speed": jitter(rng, 11.0 + 0.3 * route_bucket, 3.0, 6.0, 20.0),
            "passing_gap": float(target_lane_front_offset - slow_front_offset),
        }
        return "passing", params

    behavior = FOLLOW_BEHAVIORS[route_bucket % len(FOLLOW_BEHAVIORS)]
    params = {
        "template_name": "straight_follow",
        "behavior_type": behavior,
        "ego_lane": 1,
        "ego_speed": ego_speed,
        "front_offset": jitter(rng, 24.0 + 3.0 * route_bucket + 0.8 * init_bucket, 4.0, 14.0, 48.0),
        "front_initial_speed": jitter(rng, 13.0 + 0.4 * route_bucket, 3.0, 8.0, 22.0),
        "trigger_distance": jitter(rng, 15.0 + 2.0 * route_bucket, 4.0, 8.0, 32.0),
    }
    if behavior == "sudden_decel":
        params["decel_target_speed"] = jitter(rng, 3.5 + 0.15 * init_bucket, 1.5, 1.0, 8.0)
    elif behavior == "speed_oscillation":
        params["oscillation_base_speed"] = jitter(rng, 8.0 + 0.25 * init_bucket, 2.0, 5.0, 14.0)
        params["oscillation_amplitude"] = jitter(rng, 3.5, 1.0, 2.0, 7.0)
        params["oscillation_period"] = jitter(rng, 3.0, 0.8, 1.0, 5.0)
    elif behavior == "hard_brake":
        params["hard_brake_strength"] = jitter(rng, 0.85, 0.12, 0.6, 1.0)
        params["hard_brake_duration"] = jitter(rng, 1.2 + 0.08 * init_bucket, 0.45, 0.5, 2.5)
        params["recovery_speed"] = jitter(rng, 4.5, 1.5, 2.0, 8.0)
    return "straight_follow", params


def load_safebench_cases(args) -> List[Dict]:
    scenario_ids = parse_int_set(args.scenario_ids)
    route_ids = parse_int_set(args.route_ids)
    path = Path(args.scenario_type_json)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    grouped: Dict[Tuple[int, int], List[Dict]] = {}
    for row in rows:
        scenario_id = int(row["scenario_id"])
        route_id = int(row["route_id"])
        if scenario_ids is not None and scenario_id not in scenario_ids:
            continue
        if route_ids is not None and route_id not in route_ids:
            continue
        grouped.setdefault((scenario_id, route_id), []).append(row)

    selected = []
    routes_by_scenario: Dict[int, List[int]] = {}
    for scenario_id, route_id in sorted(grouped):
        routes_by_scenario.setdefault(scenario_id, []).append(route_id)
    allowed_routes = set()
    for scenario_id, route_list in routes_by_scenario.items():
        route_list = sorted(route_list)
        if int(args.max_routes_per_scenario) > 0:
            route_list = route_list[: int(args.max_routes_per_scenario)]
        allowed_routes.update((scenario_id, route_id) for route_id in route_list)
    for key in sorted(allowed_routes):
        case_rows = sorted(grouped[key], key=lambda item: (parameter_index(item.get("parameters"), item.get("data_id", 0)), int(item.get("data_id", 0))))
        max_inits = int(args.max_inits_per_route)
        selected.extend(case_rows if max_inits <= 0 else case_rows[:max_inits])
    return selected


def build_scene_specs(args) -> List[Dict]:
    scenes = []
    for scene_id, row in enumerate(load_safebench_cases(args), start=1):
        scenario_id = int(row["scenario_id"])
        route_id = int(row["route_id"])
        init_id = parameter_index(row.get("parameters"), int(row.get("data_id", scene_id)))
        template_name, params = synthetic_params_from_safebench_case(scenario_id, route_id, init_id, args)
        scenario_name = f"scenario_{scenario_id:02d}"
        scenes.append({
            "scene_id": scene_id,
            "data_id": int(row.get("data_id", scene_id - 1)),
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "safebench_scenario_name": SAFEBENCH_SCENARIO_NAMES.get(scenario_id, scenario_name),
            "route_id": route_id,
            "init_id": init_id,
            "parameters": parameter_label(row.get("parameters")),
            "seed": int(args.seed + scenario_id * 100000 + route_id * 1000 + init_id),
            "template_name": template_name,
            "behavior_type": params["behavior_type"],
            "decoded_params": params,
        })
    return scenes


def write_csv(path: Path, records: Iterable[Dict]) -> None:
    records = list(records)
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_eval(args, output_dir: Path, ego_items: List[Tuple[str, Path, str]], scenes: List[Dict]) -> pd.DataFrame:
    action_adapter = ActionAdapter({})
    ego_action = fixed_ego_action()
    records = []
    print("=" * 80)
    print("SafeBench-style LC ego sweep evaluation")
    print(f"  ego_models:            {len(ego_items)}")
    print(f"  scenario_type_json:    {args.scenario_type_json}")
    print(f"  scenario_ids:          {args.scenario_ids}")
    print(f"  cases:                 {len(scenes)}")
    print(f"  total_episodes:        {len(ego_items) * len(scenes)}")
    print(f"  output_dir:            {output_dir}")
    print("=" * 80)

    for model_index, (ego_label, ego_path, model_group) in enumerate(ego_items, start=1):
        ego_policy = load_ego_policy(ego_path, no_cuda=args.no_cuda, seed=args.seed)
        print(f"[{model_index:02d}/{len(ego_items):02d}] {ego_label} {ego_path}", flush=True)
        for scene_index, scene_spec in enumerate(scenes, start=1):
            scene_spec = dict(scene_spec)
            scene_spec["model_label"] = ego_label
            render_limit = int(getattr(args, "render_first_n_per_model", 1))
            args._render_current_episode = bool(args.render and (render_limit <= 0 or scene_index <= render_limit))
            template = make_template(scene_spec["template_name"], route_length=args.route_length)
            metrics = run_episode(template, scene_spec, ego_policy, ego_action, args, action_adapter)
            low_speed_ratio = float(metrics.get("low_speed_ratio", 1.0))
            progress_auc = float(metrics.get("progress_auc", 0.0))
            combined_efficiency = 0.8 * progress_auc + 0.2 * (1.0 - low_speed_ratio)
            row = {
                "model_group": model_group,
                "model_label": ego_label,
                "checkpoint_step": checkpoint_step(ego_path),
                "ego_checkpoint": str(ego_path),
                "scene_id": int(scene_spec["scene_id"]),
                "data_id": int(scene_spec["data_id"]),
                "scenario_id": int(scene_spec["scenario_id"]),
                "scenario_name": scene_spec["scenario_name"],
                "safebench_scenario_name": scene_spec["safebench_scenario_name"],
                "route_id": int(scene_spec["route_id"]),
                "init_id": int(scene_spec["init_id"]),
                "parameters": scene_spec["parameters"],
                "seed": int(scene_spec["seed"]),
                "template_name": scene_spec["template_name"],
                "behavior_type": scene_spec["behavior_type"],
                "decoded_params": json.dumps(scene_spec["decoded_params"], ensure_ascii=False, sort_keys=True),
                "combined_efficiency": float(combined_efficiency),
                "ego_reward": float(metrics.get("episode_reward", 0.0)),
                "crash": bool(metrics.get("collision", False)),
                "route_completion": float(metrics.get("route_completion", 0.0)),
                "mean_speed": float(metrics.get("average_speed", 0.0)),
                "ego_route_distance_m": float(metrics.get("driven_distance", 0.0)),
                **metrics,
            }
            records.append(row)
            if scene_index % max(1, len(scenes) // 4) == 0:
                print(
                    f"  scenes {scene_index:04d}/{len(scenes)} "
                    f"scenario={row['scenario_name']} route={row['route_id']} init={row['init_id']} "
                    f"eff={row['combined_efficiency']:.3f} "
                    f"crash={row['crash']} route={row['route_completion']:.3f}",
                    flush=True,
                )

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "combined_episode_log.csv", index=False)
    return df


def format_mean_var(mean_value, var_value) -> str:
    if pd.isna(mean_value):
        return ""
    return f"{float(mean_value):.4f} ± {float(var_value):.4f}"


def scenario_mean_variance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model_group", "model_label", "checkpoint_step", "scenario_id", "scenario_name", "safebench_scenario_name"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["episodes"] = int(len(sub))
        for metric, label in METRICS:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_var"] = float(values.var(ddof=0)) if len(values) else np.nan
            row[f"{label}_mean_var"] = format_mean_var(row[f"{metric}_mean"], row[f"{metric}_var"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["checkpoint_step", "scenario_name"])


def route_mean_variance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model_group", "model_label", "checkpoint_step", "scenario_id", "scenario_name", "route_id"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["episodes"] = int(len(sub))
        for metric, label in METRICS:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_var"] = float(values.var(ddof=0)) if len(values) else np.nan
            row[f"{label}_mean_var"] = format_mean_var(row[f"{metric}_mean"], row[f"{metric}_var"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["checkpoint_step", "scenario_id", "route_id"])


def model_mean_variance(df: pd.DataFrame) -> pd.DataFrame:
    scenario_table = scenario_mean_variance(df)
    rows = []
    group_cols = ["model_group", "model_label", "checkpoint_step"]
    for keys, sub in scenario_table.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["scenario_count"] = int(len(sub))
        row["episode_count"] = int(sub["episodes"].sum())
        for metric, label in METRICS:
            mean_col = f"{metric}_mean"
            var_col = f"{metric}_var"
            row[f"{metric}_mean"] = float(pd.to_numeric(sub[mean_col], errors="coerce").mean())
            row[f"{metric}_var"] = float(pd.to_numeric(sub[var_col], errors="coerce").mean())
            row[f"{label}_mean_var"] = format_mean_var(row[f"{metric}_mean"], row[f"{metric}_var"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("checkpoint_step")


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def save_overall_bar(plt, table: pd.DataFrame, metric: str, title: str, output_path: Path) -> bool:
    mean_col = f"{metric}_mean"
    if mean_col not in table.columns:
        return False
    fig, ax = plt.subplots(figsize=(max(11, 0.62 * len(table)), 5.2))
    x = np.arange(len(table))
    colors = ["#f58518" if group == "training_final" else "#4c78a8" for group in table["model_group"]]
    ax.bar(x, table[mean_col], color=colors)
    ax.set_title(f"{title} comparison")
    ax.set_xlabel("Ego model")
    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels(table["model_label"], rotation=45, ha="right")
    for idx, value in enumerate(table[mean_col]):
        if pd.notna(value):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def save_scenario_grouped_bar(plt, scenario_table: pd.DataFrame, metric: str, title: str, output_path: Path) -> bool:
    mean_col = f"{metric}_mean"
    if mean_col not in scenario_table.columns:
        return False
    pivot = scenario_table.pivot_table(
        index="scenario_name",
        columns="model_label",
        values=mean_col,
        aggfunc="mean",
    ).sort_index()
    model_order = (
        scenario_table[["model_label", "checkpoint_step"]]
        .drop_duplicates()
        .sort_values("checkpoint_step")["model_label"]
        .tolist()
    )
    pivot = pivot[[label for label in model_order if label in pivot.columns]]
    fig, ax = plt.subplots(figsize=(max(12, 0.75 * len(pivot.columns)), 5.8))
    x = np.arange(len(pivot.index))
    width = min(0.82 / max(len(pivot.columns), 1), 0.08)
    for idx, column in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2.0) * width
        color = "#f58518" if "train" in str(column) else None
        ax.bar(x + offset, pivot[column], width=width, label=column, color=color)
    ax.set_title(f"{title} by SafeBench scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(name).replace("_", " ") for name in pivot.index], rotation=20, ha="right")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def save_by_scenario_facets(plt, scenario_table: pd.DataFrame, metric: str, title: str, output_path: Path) -> bool:
    mean_col = f"{metric}_mean"
    if mean_col not in scenario_table.columns:
        return False
    model_order = (
        scenario_table[["model_label", "checkpoint_step"]]
        .drop_duplicates()
        .sort_values("checkpoint_step")["model_label"]
        .tolist()
    )
    scenario_names = sorted(str(item) for item in scenario_table["scenario_name"].dropna().unique())
    ncols = 2
    nrows = int(np.ceil(max(len(scenario_names), 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, max(4.8 * nrows, 5.0)), squeeze=False)
    for ax, scenario_name in zip(axes.flatten(), scenario_names):
        sub = scenario_table[scenario_table["scenario_name"] == scenario_name].copy()
        sub["model_label"] = pd.Categorical(sub["model_label"], categories=model_order, ordered=True)
        sub = sub.sort_values("model_label")
        x = np.arange(len(sub))
        colors = ["#f58518" if group == "training_final" else "#4c78a8" for group in sub["model_group"]]
        ax.bar(x, sub[mean_col], color=colors)
        ax.set_title(scenario_name.replace("_", " "))
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model_label"], rotation=45, ha="right", fontsize=7)
        for idx, value in enumerate(sub[mean_col]):
            if pd.notna(value):
                ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=6)
    for ax in axes.flatten()[len(scenario_names):]:
        ax.axis("off")
    fig.suptitle(f"{title} under each SafeBench scenario", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def summarize_and_plot(output_dir: Path) -> None:
    df = pd.read_csv(output_dir / "combined_episode_log.csv")
    for metric, _ in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    scenario_table = scenario_mean_variance(df)
    route_table = route_mean_variance(df)
    model_table = model_mean_variance(df)
    scenario_table.to_csv(output_dir / "scenario_mean_variance.csv", index=False)
    route_table.to_csv(output_dir / "route_mean_variance.csv", index=False)
    model_table.to_csv(output_dir / "model_mean_variance_numeric.csv", index=False)

    formatted_cols = ["model_group", "model_label", "checkpoint_step", "scenario_count", "episode_count"]
    formatted_cols.extend([f"{label}_mean_var" for _, label in METRICS])
    model_table[formatted_cols].to_csv(output_dir / "model_comparison_mean_plus_variance.csv", index=False)

    scenario_cols = ["model_group", "model_label", "checkpoint_step", "scenario_id", "scenario_name", "safebench_scenario_name", "episodes"]
    scenario_cols.extend([f"{label}_mean_var" for _, label in METRICS])
    scenario_table[scenario_cols].to_csv(output_dir / "scenario_comparison_mean_plus_variance.csv", index=False)

    route_cols = ["model_group", "model_label", "checkpoint_step", "scenario_id", "scenario_name", "route_id", "episodes"]
    route_cols.extend([f"{label}_mean_var" for _, label in METRICS])
    route_table[route_cols].to_csv(output_dir / "route_comparison_mean_plus_variance.csv", index=False)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    for metric, title in BAR_METRICS:
        save_overall_bar(plt, model_table, metric, title, figures_dir / f"overall_{metric}_bar.png")
        save_scenario_grouped_bar(plt, scenario_table, metric, title, figures_dir / f"scenario_mean_{metric}_bar.png")
        save_by_scenario_facets(plt, scenario_table, metric, title, figures_dir / f"by_scenario_{metric}_bar.png")
    for metric, title in [("route_completion", "Road completion"), ("crash", "Ego crash rate"), ("episode_reward", "Average reward")]:
        save_overall_bar(plt, model_table, metric, title, figures_dir / f"overall_{metric}_bar.png")
        save_scenario_grouped_bar(plt, scenario_table, metric, title, figures_dir / f"scenario_mean_{metric}_bar.png")
        save_by_scenario_facets(plt, scenario_table, metric, title, figures_dir / f"by_scenario_{metric}_bar.png")

    print(f"Output dir: {output_dir}")
    print(f"Episode log: {output_dir / 'combined_episode_log.csv'}")
    print(f"Model table: {output_dir / 'model_comparison_mean_plus_variance.csv'}")
    print(f"Scenario table: {output_dir / 'scenario_comparison_mean_plus_variance.csv'}")
    print(f"Route table: {output_dir / 'route_comparison_mean_plus_variance.csv'}")
    print(f"Figures dir: {figures_dir}")


def main():
    args = parse_args()
    output_dir = make_output_dir(args)
    if not args.summarize_only:
        ego_items = resolve_ego_checkpoints(args)
        scenes = build_scene_specs(args)
        if args.dry_run:
            print("=" * 80)
            print("SafeBench-style LC ego sweep dry run")
            print(f"  output_dir: {output_dir}")
            print(f"  scenario_type_json: {args.scenario_type_json}")
            print(f"  scenario_ids: {args.scenario_ids}")
            print(f"  route_ids: {args.route_ids or 'all'}")
            print(f"  ego_models: {len(ego_items)}")
            for label, path, group in ego_items:
                print(f"    {label:16s} {group:15s} {path}")
            print(f"  cases: {len(scenes)}")
            counts = (
                pd.DataFrame(scenes)
                .groupby(["scenario_id", "scenario_name", "route_id"], dropna=False)
                .size()
                .reset_index(name="init_count")
            )
            print(counts.to_string(index=False))
            return
        write_csv(output_dir / "fixed_scene_params.csv", [
            {
                "scene_id": scene["scene_id"],
                "data_id": scene["data_id"],
                "scenario_id": scene["scenario_id"],
                "scenario_name": scene["scenario_name"],
                "safebench_scenario_name": scene["safebench_scenario_name"],
                "route_id": scene["route_id"],
                "init_id": scene["init_id"],
                "parameters": scene["parameters"],
                "seed": scene["seed"],
                "template_name": scene["template_name"],
                "behavior_type": scene["behavior_type"],
                "decoded_params": json.dumps(scene["decoded_params"], ensure_ascii=False, sort_keys=True),
            }
            for scene in scenes
        ])
        manifest = {
            "training_ego_checkpoint": str(args.training_ego_checkpoint),
            "finetune_ego_dir": str(args.finetune_ego_dir),
            "checkpoint_interval": int(args.checkpoint_interval),
            "include_final": bool(args.include_final),
            "scenario_type_json": str(args.scenario_type_json),
            "scenario_ids": str(args.scenario_ids),
            "route_ids": str(args.route_ids),
            "max_routes_per_scenario": int(args.max_routes_per_scenario),
            "max_inits_per_route": int(args.max_inits_per_route),
            "max_steps": int(args.max_steps),
            "route_length": float(args.route_length),
            "dt": float(args.dt),
            "seed": int(args.seed),
            "ego_models": [{"label": label, "checkpoint": str(path), "group": group} for label, path, group in ego_items],
            "case_count": len(scenes),
        }
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        run_eval(args, output_dir, ego_items, scenes)
    summarize_and_plot(output_dir)


if __name__ == "__main__":
    main()
