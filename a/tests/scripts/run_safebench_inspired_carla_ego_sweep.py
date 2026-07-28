#!/usr/bin/env python3
"""Run SafeBench-inspired route/init ego sweep in this project's CARLA env."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = MODEL_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from carla_evolution.scripts import finetune_ego_enhanced_poet as finetune


DEFAULT_TRAINING_EGO = MODEL_ROOT / "results" / "joint_Jul_01_14_59_59" / "models" / "ego" / "checkpoint-62710.pt"
DEFAULT_FINETUNE_DIR = MODEL_ROOT / "results" / "ego_enhanced_poet_finetune" / "poet_finetune_Jul_21_13_29_28-30-10" / "models" / "ego"
DEFAULT_SCENARIO_TYPE_JSON = Path("/home/chenyuanwan/download/SafeBench-main/safebench/scenario/config/scenario_type/adv_init_state.json")
DEFAULT_CONFIG = MODEL_ROOT / "configs" / "carla_0913.yaml"
DEFAULT_ADV_MODEL_DIR = MODEL_ROOT / "results" / "joint_Jul_01_14_59_59" / "models" / "adv"

SCENE_BY_ROUTE = [
    ("follow_straight", "sudden_decel"),
    ("passing", "slow_front_overtake"),
    ("lane_change", "right_escape"),
    ("cut_in", "cut_in_recovery"),
    ("follow_straight", "speed_oscillation"),
    ("passing", "slow_front_overtake"),
    ("lane_change", "right_escape"),
    ("cut_in", "cut_in_recovery"),
    ("follow_straight", "hard_brake"),
    ("cut_in", "cut_in_recovery"),
]

METRICS = [
    ("route_completion", "Road completion"),
    ("crash", "Ego crash rate"),
    ("combined_efficiency", "Traffic efficiency"),
    ("ego_reward", "Ego reward"),
    ("episode_reward", "Average reward"),
    ("mean_speed", "Average speed"),
    ("ego_route_distance_m", "Driven distance"),
    ("ego_overtake_success_flag", "Overtake success rate"),
]

BAR_METRICS = [
    ("combined_efficiency", "Traffic efficiency"),
    ("ego_route_distance_m", "Driven distance"),
    ("mean_speed", "Average speed"),
]

SCENARIO_FAMILY_METRICS = [
    ("crash", "Collision rate", "crash_rate"),
    ("route_completion", "Road completion", "route_completion"),
    ("ego_route_distance_m", "Driven distance", "ego_route_distance_m"),
    ("mean_speed", "Average speed", "mean_speed"),
    ("combined_efficiency", "Traffic efficiency", "combined_efficiency"),
    ("ego_reward", "Ego reward", "ego_reward"),
    ("episode_reward", "Average reward", "episode_reward"),
]

SCENARIO_FAMILY_ORDER = ["follow_straight", "passing", "lane_change", "cut_in"]

BEHAVIOR_TYPES = [
    "sudden_decel",
    "speed_oscillation",
    "hard_brake",
    "slow_front_overtake",
    "right_escape",
    "cut_in_recovery",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scenario-type-json", default=str(DEFAULT_SCENARIO_TYPE_JSON))
    parser.add_argument("--scenario-ids", default="2")
    parser.add_argument("--route-ids", default="")
    parser.add_argument("--max-routes-per-scenario", type=int, default=0)
    parser.add_argument("--max-inits-per-route", type=int, default=10, help="0 keeps all init cases per route.")
    parser.add_argument("--training-ego-checkpoint", default=str(DEFAULT_TRAINING_EGO))
    parser.add_argument("--training-ego-label", default="")
    parser.add_argument("--finetune-ego-dir", default=str(DEFAULT_FINETUNE_DIR))
    parser.add_argument("--finetune-ego-checkpoints", default="")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--include-final", action="store_true", default=True)
    parser.add_argument("--no-include-final", action="store_false", dest="include_final")
    parser.add_argument("--adv-model-dir", default=str(DEFAULT_ADV_MODEL_DIR))
    parser.add_argument("--adv-step", type=int, default=0, help="0 selects latest paired MAPPO checkpoint under --adv-model-dir.")
    parser.add_argument("--num-adv", type=int, default=3)
    parser.add_argument("--num-natural", type=int, default=0)
    parser.add_argument("--hdv-model", default="")
    parser.add_argument("--hdv-action", default="keep_lane")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--route-length", type=float, default=200.0)
    parser.add_argument("--target-speed-cruise", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=669)
    parser.add_argument("--torch-seed", type=int, default=669)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--carla-rpc-timeout", type=float, default=180.0)
    parser.add_argument("--purge-existing-actors-on-reset", action="store_true", default=False)
    parser.add_argument("--cleanup-destroy-mode", choices=["sequential", "batch"], default="sequential")
    parser.add_argument("--case-cooldown", type=float, default=0.2, help="Seconds to pause after closing each CARLA case.")
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--summarize-only", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
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


def load_ego_policy(path: Path, env, no_cuda: bool, seed: int, EgoPPOAdapter):
    if str(path).endswith(".zip"):
        from carla_evolution.tests.envs.sb3_ego_adapter import SB3ContinuousEgoAsDiscretePolicy

        return SB3ContinuousEgoAsDiscretePolicy(path)
    ego = EgoPPOAdapter.from_config({"use_cuda": not bool(no_cuda), "seed": int(seed)}, env.n_s, env.n_a)
    ego.load(str(path))
    return ego


def parse_int_set(value: str) -> Optional[set]:
    items = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return {int(item) for item in items} if items else None


def parameter_label(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def parameter_index(value, fallback: int) -> int:
    matches = re.findall(r"(\d+)", parameter_label(value))
    return int(matches[-1]) if matches else int(fallback)


def load_safebench_cases(args) -> List[Dict]:
    scenario_ids = parse_int_set(args.scenario_ids)
    route_ids = parse_int_set(args.route_ids)
    with Path(args.scenario_type_json).open("r", encoding="utf-8") as f:
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
    allowed = set()
    for scenario_id, route_list in routes_by_scenario.items():
        route_list = sorted(route_list)
        if int(args.max_routes_per_scenario) > 0:
            route_list = route_list[: int(args.max_routes_per_scenario)]
        allowed.update((scenario_id, route_id) for route_id in route_list)
    for key in sorted(allowed):
        case_rows = sorted(
            grouped[key],
            key=lambda item: (parameter_index(item.get("parameters"), item.get("data_id", 0)), int(item.get("data_id", 0))),
        )
        max_inits = int(args.max_inits_per_route)
        selected.extend(case_rows if max_inits <= 0 else case_rows[:max_inits])
    return selected


def latest_mappo_step(model_dir: Path) -> int:
    steps = []
    for actor in model_dir.glob("actor_*.pt"):
        match = re.fullmatch(r"actor_(\d+)\.pt", actor.name)
        if not match:
            continue
        step = int(match.group(1))
        if (model_dir / f"critic_{step}.pt").exists():
            steps.append(step)
    if not steps:
        raise FileNotFoundError(f"No MAPPO actor/critic checkpoints found under {model_dir}")
    return max(steps)


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%b_%d_%H_%M_%S")
        output_dir = MODEL_ROOT / "tests" / "results" / f"safebench_inspired_carla_ego_sweep_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    return output_dir


def case_to_spec(case: Dict, adv_step: int, args):
    scenario_id = int(case["scenario_id"])
    route_id = int(case["route_id"])
    init_id = parameter_index(case.get("parameters"), int(case.get("data_id", 0)))
    template, behavior_type = SCENE_BY_ROUTE[route_id % len(SCENE_BY_ROUTE)]
    rng = np.random.RandomState(int(args.seed) + scenario_id * 100000 + route_id * 1000 + init_id)
    spec = finetune.make_env_spec(
        env_id=int(case.get("data_id", 0)),
        template=template,
        adv_step=int(adv_step),
        num_adv=int(args.num_adv),
        rng=rng,
    )
    spec.num_natural = int(args.num_natural)
    spec.route_length = float(args.route_length)
    spec.target_speed_cruise = float(args.target_speed_cruise)
    spec.offsets = [float(item + rng.uniform(-4.0, 4.0) + 0.8 * init_id) for item in spec.offsets]
    spec.lane_offsets = [int(np.clip((lane + route_id + init_id) % 3, 0, 2)) for lane in spec.lane_offsets]
    spec.mutation_type = f"safebench_case_s{scenario_id}_r{route_id}_i{init_id}_{behavior_type}"
    return spec


def behavior_from_spec(spec) -> str:
    mutation = str(getattr(spec, "mutation_type", ""))
    for behavior in BEHAVIOR_TYPES:
        if mutation.endswith(f"_{behavior}"):
            return behavior
    return ""


def env_args(args, case: Dict):
    route_id = int(case["route_id"])
    init_id = parameter_index(case.get("parameters"), int(case.get("data_id", 0)))
    return SimpleNamespace(
        backend="carla",
        config=args.config,
        max_steps=int(args.max_steps),
        spawn_start_index=int(route_id * 3 + init_id),
        carla_rpc_timeout=float(args.carla_rpc_timeout),
        render=bool(args.render),
        hdv_action=args.hdv_action,
        _hdv_policy=None,
        randomize_scenarios=False,
        relative_offset_jitter=0.0,
        randomize_lane_offsets=False,
        randomize_lane_offset_min=0,
        randomize_lane_offset_max=2,
        randomize_spawn_start_index=False,
        shuffle_spawn_points=False,
        low_speed_alpha=0.5,
        low_risk_threshold=0.25,
        front_slow_alpha=0.6,
        front_slow_gap=35.0,
        no_cuda=bool(args.no_cuda),
        torch_seed=int(args.torch_seed),
        min_num_adv=int(args.num_adv),
        max_num_adv=int(args.num_adv),
        purge_existing_actors_on_reset=bool(args.purge_existing_actors_on_reset),
        cleanup_destroy_mode=str(args.cleanup_destroy_mode),
    )


def write_csv(path: Path, records: Iterable[Dict]) -> None:
    records = list(records)
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def episode_row(model_label, model_group, ego_checkpoint, case, spec, metrics, train_mod):
    row = {
        "model_group": model_group,
        "model_label": model_label,
        "checkpoint_step": checkpoint_step(Path(ego_checkpoint)),
        "ego_checkpoint": str(ego_checkpoint),
        "data_id": int(case.get("data_id", 0)),
        "scenario_id": int(case["scenario_id"]),
        "scenario_name": f"scenario_{int(case['scenario_id']):02d}",
        "route_id": int(case["route_id"]),
        "init_id": parameter_index(case.get("parameters"), int(case.get("data_id", 0))),
        "parameters": parameter_label(case.get("parameters")),
        "template": spec.template,
        "scenario_family": spec.template,
        "behavior_type": behavior_from_spec(spec),
        "seed": int(case.get("seed", 0)),
        "adv_step": int(spec.adv_step),
        "combined_efficiency": float(metrics.get("combined_efficiency", 0.0)),
        "ego_reward": float(metrics.get("ego_reward", metrics.get("episode_reward", 0.0))),
        "traffic_efficiency": finetune.efficiency_from_metrics(metrics),
        "progress_auc": float(metrics.get("progress_auc", 0.0)),
        "route_speed_efficiency": finetune.route_speed_efficiency_from_metrics(metrics),
        "route_completion": float(metrics.get("route_completion", 0.0)),
        "route_progress_m": float(metrics.get("route_progress_m", finetune.route_progress_from_mapping(metrics))),
        "ego_route_distance_m": float(metrics.get("ego_route_distance_m", metrics.get("route_progress_m", 0.0))),
        "completion_horizon_m": float(metrics.get("route_completion_distance", 0.0)),
        "elapsed_steps": int(metrics.get("steps", 0)),
        "elapsed_time_s": float(metrics.get("elapsed_time_s", float(metrics.get("steps", 0)) * finetune.metrics_dt(metrics))),
        "fixed_delta_seconds": finetune.metrics_dt(metrics),
        "mean_speed": float(metrics.get("average_speed", 0.0)),
        "max_speed": float(metrics.get("max_speed", metrics.get("average_speed", 0.0))),
        "min_speed": float(metrics.get("min_speed", metrics.get("average_speed", 0.0))),
        "low_speed_ratio": float(metrics.get("low_speed_ratio", 1.0)),
        "low_risk_steps": int(metrics.get("low_risk_steps", 0)),
        "low_speed_steps": int(metrics.get("low_speed_steps", 0)),
        "front_slow_flag": bool(metrics.get("front_slow_flag", False)),
        "front_slow_steps": int(metrics.get("front_slow_steps", 0)),
        "front_slow_ratio": float(metrics.get("front_slow_ratio", 0.0)),
        "safe_lane_change_opportunity_flag": bool(metrics.get("safe_lane_change_opportunity_flag", False)),
        "safe_lane_change_opportunity_steps": int(metrics.get("safe_lane_change_opportunity_steps", 0)),
        "safe_lane_change_opportunity_ratio": float(metrics.get("safe_lane_change_opportunity_ratio", 0.0)),
        "ego_lane_change_action_flag": bool(metrics.get("ego_lane_change_action_flag", False)),
        "ego_lane_change_action_steps": int(metrics.get("ego_lane_change_action_steps", 0)),
        "ego_lane_change_action_ratio": float(metrics.get("ego_lane_change_action_ratio", 0.0)),
        "ego_lane_change_success_flag": bool(metrics.get("ego_lane_change_success_flag", False)),
        "ego_lane_change_success_steps": int(metrics.get("ego_lane_change_success_steps", 0)),
        "ego_overtake_success_flag": bool(metrics.get("ego_overtake_success_flag", False)),
        "ego_overtake_success_steps": int(metrics.get("ego_overtake_success_steps", 0)),
        "ego_lane_change_intent_flag": bool(metrics.get("ego_lane_change_intent_flag", False)),
        "ego_lane_change_intent_steps": int(metrics.get("ego_lane_change_intent_steps", 0)),
    }
    row.update(train_mod.episode_log_row(0, metrics, {}))
    return row


def format_mean_var(mean_value, var_value) -> str:
    if pd.isna(mean_value):
        return ""
    return f"{float(mean_value):.4f} ± {float(var_value):.4f}"


def grouped_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["episodes"] = int(len(sub))
        for metric, label in METRICS:
            if metric not in sub.columns:
                continue
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_var"] = float(values.var(ddof=0)) if len(values) else np.nan
            row[f"{label}_mean_var"] = format_mean_var(row[f"{metric}_mean"], row[f"{metric}_var"])
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_family_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "scenario_family" not in df.columns:
        df = df.copy()
        df["scenario_family"] = df.get("template", df.get("scenario_name", "scenario"))
    group_cols = ["model_group", "model_label", "checkpoint_step", "scenario_family"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["episodes"] = int(len(sub))
        for metric, label, _slug in SCENARIO_FAMILY_METRICS:
            if metric not in sub.columns:
                continue
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_var"] = float(values.var(ddof=0)) if len(values) else np.nan
            row[f"{label}_mean_var"] = format_mean_var(row[f"{metric}_mean"], row[f"{metric}_var"])
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    order = {name: idx for idx, name in enumerate(SCENARIO_FAMILY_ORDER)}
    table["_scenario_order"] = table["scenario_family"].map(order).fillna(len(order))
    return table.sort_values(["checkpoint_step", "_scenario_order", "scenario_family"]).drop(columns=["_scenario_order"])


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.grid": True, "grid.alpha": 0.25})
    return plt


def save_bar(plt, table: pd.DataFrame, metric: str, title: str, path: Path):
    mean_col = f"{metric}_mean"
    if mean_col not in table.columns:
        return
    table = table.sort_values("checkpoint_step")
    fig, ax = plt.subplots(figsize=(max(11, 0.65 * len(table)), 5.2))
    x = np.arange(len(table))
    colors = ["#f58518" if group == "training_final" else "#4c78a8" for group in table["model_group"]]
    ax.bar(x, table[mean_col], color=colors)
    ax.set_title(f"{title} comparison")
    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels(table["model_label"], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_scenario_family_grouped_bar(plt, table: pd.DataFrame, metric: str, title: str, path: Path):
    mean_col = f"{metric}_mean"
    if mean_col not in table.columns or table.empty:
        return
    scenarios = [item for item in SCENARIO_FAMILY_ORDER if item in set(table["scenario_family"])]
    scenarios.extend(sorted(set(table["scenario_family"]) - set(scenarios)))
    labels = (
        table[["model_label", "checkpoint_step"]]
        .drop_duplicates()
        .sort_values("checkpoint_step")["model_label"]
        .tolist()
    )
    pivot = table.pivot_table(index="scenario_family", columns="model_label", values=mean_col, aggfunc="mean")
    pivot = pivot.reindex(scenarios)
    pivot = pivot[[label for label in labels if label in pivot.columns]]
    fig, ax = plt.subplots(figsize=(max(13.5, 0.9 * len(pivot.columns)), 6.2))
    x = np.arange(len(pivot.index))
    width = min(0.82 / max(len(pivot.columns), 1), 0.075)
    for idx, column in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2.0) * width
        color = "#f58518" if "train" in str(column) else None
        ax.bar(x + offset, pivot[column], width=width, label=column, color=color)
    ax.set_title(f"{title} by scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(name).replace("_", " ") for name in pivot.index], rotation=15, ha="right")
    ax.legend(loc="best", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_by_scenario_family_facets(plt, table: pd.DataFrame, metric: str, title: str, path: Path):
    mean_col = f"{metric}_mean"
    if mean_col not in table.columns or table.empty:
        return
    labels = (
        table[["model_label", "checkpoint_step"]]
        .drop_duplicates()
        .sort_values("checkpoint_step")["model_label"]
        .tolist()
    )
    scenarios = [item for item in SCENARIO_FAMILY_ORDER if item in set(table["scenario_family"])]
    scenarios.extend(sorted(set(table["scenario_family"]) - set(scenarios)))
    ncols = 2
    nrows = int(np.ceil(max(len(scenarios), 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, max(4.8 * nrows, 5.0)), squeeze=False)
    for ax, scenario in zip(axes.flatten(), scenarios):
        sub = table[table["scenario_family"] == scenario].copy()
        sub["model_label"] = pd.Categorical(sub["model_label"], categories=labels, ordered=True)
        sub = sub.sort_values("model_label")
        colors = ["#f58518" if group == "training_final" else "#4c78a8" for group in sub["model_group"]]
        x = np.arange(len(sub))
        ax.bar(x, sub[mean_col], color=colors)
        ax.set_title(str(scenario).replace("_", " "))
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model_label"], rotation=45, ha="right", fontsize=7)
    for ax in axes.flatten()[len(scenarios):]:
        ax.axis("off")
    fig.suptitle(f"{title} under each scenario", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize(output_dir: Path):
    df = pd.read_csv(output_dir / "combined_episode_log.csv")
    for metric, _ in METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    scenario_table = grouped_stats(df, ["model_group", "model_label", "checkpoint_step", "scenario_id", "scenario_name"])
    route_table = grouped_stats(df, ["model_group", "model_label", "checkpoint_step", "scenario_id", "scenario_name", "route_id"])
    model_table = grouped_stats(df, ["model_group", "model_label", "checkpoint_step"])
    scenario_family_table = scenario_family_stats(df)
    scenario_table.to_csv(output_dir / "scenario_mean_variance.csv", index=False)
    route_table.to_csv(output_dir / "route_mean_variance.csv", index=False)
    model_table.to_csv(output_dir / "model_mean_variance_numeric.csv", index=False)
    scenario_family_table.to_csv(output_dir / "scenario_family_mean_variance.csv", index=False)
    fmt_cols = ["model_group", "model_label", "checkpoint_step", "episodes"] + [f"{label}_mean_var" for _, label in METRICS if f"{label}_mean_var" in model_table.columns]
    model_table[fmt_cols].to_csv(output_dir / "model_comparison_mean_plus_variance.csv", index=False)
    family_cols = ["model_group", "model_label", "checkpoint_step", "scenario_family", "episodes"]
    family_cols.extend([f"{label}_mean_var" for _, label, _slug in SCENARIO_FAMILY_METRICS if f"{label}_mean_var" in scenario_family_table.columns])
    scenario_family_table[family_cols].to_csv(output_dir / "scenario_family_comparison_mean_plus_variance.csv", index=False)
    plt = setup_matplotlib()
    for metric, title in BAR_METRICS:
        save_bar(plt, model_table, metric, title, output_dir / "figures" / f"overall_{metric}_bar.png")
    for metric, title, slug in SCENARIO_FAMILY_METRICS:
        save_scenario_family_grouped_bar(plt, scenario_family_table, metric, title, output_dir / "figures" / f"scenario_mean_{slug}_bar.png")
        save_by_scenario_family_facets(plt, scenario_family_table, metric, title, output_dir / "figures" / f"by_scenario_{slug}_bar.png")


def main():
    args = parse_args()
    output_dir = make_output_dir(args)
    if args.summarize_only:
        summarize(output_dir)
        return

    ego_items = resolve_ego_checkpoints(args)
    cases = load_safebench_cases(args)
    adv_model_dir = Path(args.adv_model_dir)
    adv_step = int(args.adv_step) if int(args.adv_step) > 0 else latest_mappo_step(adv_model_dir)
    if args.dry_run:
        print(f"ego_models={len(ego_items)} cases={len(cases)} adv_step={adv_step} output={output_dir}")
        print(pd.DataFrame(cases).groupby(["scenario_id", "route_id"]).size().reset_index(name="init_count").to_string(index=False))
        return

    train_mod, MAPPOAgent, EgoPPOAdapter, make_env = finetune.runtime_imports()
    adv_ckpt = finetune.AdvCheckpoint(step=adv_step, model_dir=str(adv_model_dir))
    all_rows = []
    case_rows = []
    for case in cases:
        spec = case_to_spec(case, adv_step, args)
        case_seed = int(args.seed + int(case["scenario_id"]) * 100000 + int(case["route_id"]) * 1000 + parameter_index(case.get("parameters"), 0))
        case["seed"] = case_seed
        case_rows.append({**case, "resolved_template": spec.template, "resolved_spec": json.dumps(asdict(spec), ensure_ascii=False)})
        ea = env_args(args, case)
        if str(args.hdv_model).strip():
            hdv_path = finetune.latest_hdv_model(MODEL_ROOT / "results") if args.hdv_model == "latest" else Path(args.hdv_model)
            ea._hdv_policy = train_mod.load_hdv_policy(str(hdv_path))
        env = make_env(finetune.env_config_from_spec(spec, ea), config_path=args.config)
        try:
            if int(args.num_adv) > 0:
                mappo = MAPPOAgent.from_config({"reward_type": "agents_rewards", "use_cuda": not args.no_cuda, "torch_seed": args.torch_seed}, env.n_s, env.n_a)
                mappo.load(adv_ckpt.model_dir, adv_ckpt.step, train_mode=False)
            else:
                mappo = None
            for model_label, ego_checkpoint, model_group in ego_items:
                ego = load_ego_policy(ego_checkpoint, env, args.no_cuda, args.torch_seed, EgoPPOAdapter)
                metrics = finetune.run_eval_episode(env, mappo, ego, train_mod, ea, case_seed)
                row = episode_row(model_label, model_group, ego_checkpoint, case, spec, metrics, train_mod)
                all_rows.append(row)
                print(
                    f"{model_label} s={row['scenario_id']} r={row['route_id']} i={row['init_id']} "
                    f"eff={row['combined_efficiency']:.3f} speed={row['mean_speed']:.2f} "
                    f"dist={row['ego_route_distance_m']:.1f} crash={row.get('crash', False)}",
                    flush=True,
                )
        finally:
            env.close()
            gc.collect()
            if float(args.case_cooldown) > 0.0:
                time.sleep(float(args.case_cooldown))
        write_csv(output_dir / "combined_episode_log.csv", all_rows)
    write_csv(output_dir / "resolved_cases.csv", case_rows)
    manifest = {
        "case_source": str(args.scenario_type_json),
        "scenario_ids": args.scenario_ids,
        "route_ids": args.route_ids,
        "case_count": len(cases),
        "ego_count": len(ego_items),
        "adv_model_dir": str(adv_model_dir),
        "adv_step": adv_step,
        "mapping": "SafeBench scenario_id/route_id/init_id -> this project's CARLA spawn_start_index/relative offsets/lane offsets.",
        "args": vars(args),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    summarize(output_dir)
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
