#!/usr/bin/env python3
"""Evaluate EgoPPO checkpoints on shared randomized natural-traffic seeds."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = MODEL_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from carla_evolution.scripts import finetune_ego_enhanced_poet as finetune


DEFAULT_EGO_DIR = MODEL_ROOT / "results" / "ego_enhanced_poet_finetune" / "poet_finetune_Jul_21_13_29_28-30-10" / "models" / "ego"
DEFAULT_CONFIG = MODEL_ROOT / "configs" / "carla_0915.yaml"
DEFAULT_OUTPUT_ROOT = MODEL_ROOT / "results" / "ego_natural_seed_eval"

METRIC_KEYS = [
    "combined_efficiency",
    "traffic_efficiency",
    "progress_auc",
    "route_speed_efficiency",
    "episode_reward",
    "route_completion",
    "route_progress_m",
    "ego_route_distance_m",
    "elapsed_steps",
    "elapsed_time_s",
    "mean_speed",
    "max_speed",
    "min_speed",
    "low_speed_ratio",
    "front_slow_steps",
    "front_slow_ratio",
    "safe_lane_change_opportunity_steps",
    "safe_lane_change_opportunity_ratio",
    "ego_lane_change_action_steps",
    "ego_lane_change_action_ratio",
    "ego_lane_change_success_steps",
    "ego_overtake_success_steps",
]

PLOT_METRICS = [
    ("combined_efficiency", "Traffic efficiency"),
    ("ego_route_distance_m", "Driven distance"),
    ("mean_speed", "Average speed"),
    ("ego_crash", "Ego crash rate"),
]

REQUESTED_TABLE_METRICS = [
    ("route_completion", "road_completion"),
    ("ego_route_distance_m", "driven_distance"),
    ("mean_speed", "average_speed"),
    ("episode_reward", "average_reward"),
    ("ego_crash", "crash_rate"),
    ("combined_efficiency", "traffic_efficiency"),
]

SCENARIO_FAMILY_TABLE_METRICS = [
    ("ego_crash", "Collision rate"),
    ("route_completion", "Road completion"),
    ("ego_route_distance_m", "Driven distance"),
    ("mean_speed", "Average speed"),
    ("combined_efficiency", "Traffic efficiency"),
    ("episode_reward", "Average reward"),
]

RATE_KEYS = [
    "ego_crash",
    "crash",
    "agents_crash",
    "front_slow_flag",
    "safe_lane_change_opportunity_flag",
    "ego_lane_change_action_flag",
    "ego_lane_change_success_flag",
    "ego_overtake_success_flag",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "carla"], default="carla")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ego-dir", default=str(DEFAULT_EGO_DIR))
    parser.add_argument(
        "--ego-checkpoint",
        action="append",
        default=None,
        help="Additional EgoPPO checkpoint path to evaluate. Repeatable.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--include-final", action="store_true", default=False)
    parser.add_argument(
        "--label-with-source",
        action="store_true",
        default=False,
        help="Prefix model labels with the checkpoint's experiment directory to avoid merging identical step numbers.",
    )
    parser.add_argument("--eval-seeds", default="", help="Comma-separated scenario seeds. Overrides --seed-count when set.")
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--eval-seed-base", type=int, default=101)
    parser.add_argument("--scenario-preset", default="slow_front_left_escape")
    parser.add_argument("--num-natural", type=int, default=3)
    parser.add_argument("--spawn-start-index", type=int, default=0)
    parser.add_argument("--randomize-scenarios", action="store_true", default=False)
    parser.add_argument("--relative-offset-jitter", type=float, default=0.0)
    parser.add_argument("--randomize-lane-offsets", action="store_true", default=False)
    parser.add_argument("--randomize-lane-offset-min", type=int, default=0)
    parser.add_argument("--randomize-lane-offset-max", type=int, default=2)
    parser.add_argument(
        "--hdv-control-mode",
        choices=["traffic_manager", "fixed", "policy"],
        default="fixed",
        help="Natural-traffic control source. fixed uses this project's vehicle controller; traffic_manager uses CARLA TrafficManager.",
    )
    parser.add_argument("--hdv-model", default="", help="'latest', explicit SB3 .zip path, or empty string to use fixed HDV action.")
    parser.add_argument("--hdv-action", default="keep_lane")
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--hdv-tm-speed-difference", type=float, default=0.0)
    parser.add_argument("--hdv-tm-distance-to-leading-vehicle", type=float, default=8.0)
    parser.add_argument("--hdv-tm-auto-lane-change", action="store_true", default=True)
    parser.add_argument("--no-hdv-tm-auto-lane-change", dest="hdv_tm_auto_lane_change", action="store_false")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--carla-rpc-timeout", type=float, default=300.0)
    parser.add_argument("--torch-seed", type=int, default=669)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--purge-existing-actors-on-reset", action="store_true", default=False)
    parser.add_argument("--cleanup-destroy-mode", choices=["sequential", "batch"], default="sequential")
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--rerun-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)\.pt", path.name)
    if not match:
        raise ValueError(path.name)
    return int(match.group(1))


def checkpoint_source(path: Path) -> str:
    parts = Path(path).parts
    if "models" in parts:
        model_index = parts.index("models")
        if model_index > 0:
            return parts[model_index - 1]
    if len(Path(path).parents) >= 3:
        return Path(path).parents[2].name
    return Path(path).parent.name


def checkpoint_label(path: Path, label_with_source: bool = False) -> str:
    step = checkpoint_step(path)
    if not label_with_source:
        return f"Ego-{step}"
    source = checkpoint_source(path)
    source = source.replace("poet_finetune_", "").replace("joint_", "joint-")
    return f"{source}-Ego-{step}"


def select_checkpoints(ego_dir: Path, interval: int, include_final: bool):
    checkpoints = sorted(ego_dir.glob("checkpoint-*.pt"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-*.pt found under {ego_dir}")
    selected = []
    last_step = None
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        if last_step is None or step - last_step >= int(interval):
            selected.append(checkpoint)
            last_step = step
    if include_final and checkpoints[-1] not in selected:
        selected.append(checkpoints[-1])
    return selected


def select_all_checkpoints(args):
    selected = select_checkpoints(Path(args.ego_dir), int(args.checkpoint_interval), bool(args.include_final))
    seen = {str(path.resolve()) for path in selected}
    for item in args.ego_checkpoint or []:
        checkpoint = Path(item)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Ego checkpoint not found: {checkpoint}")
        resolved = str(checkpoint.resolve())
        if resolved not in seen:
            selected.append(checkpoint)
            seen.add(resolved)
    return sorted(selected, key=checkpoint_step)


def parse_seed_list(value: str):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def scenario_seeds(args):
    seeds = parse_seed_list(args.eval_seeds)
    if seeds:
        return seeds
    return [int(args.eval_seed_base) + idx for idx in range(int(args.seed_count))]


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%b_%d_%H_%M_%S")
        output_dir = DEFAULT_OUTPUT_ROOT / f"ego_natural_seed_eval_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_env_args(args):
    return SimpleNamespace(
        backend=args.backend,
        config=args.config,
        max_steps=int(args.max_steps),
        spawn_start_index=int(args.spawn_start_index),
        carla_rpc_timeout=float(args.carla_rpc_timeout),
        render=bool(args.render),
        hdv_control_mode=str(args.hdv_control_mode),
        hdv_action=args.hdv_action,
        _hdv_policy=None,
        traffic_manager_port=int(args.traffic_manager_port),
        hdv_tm_speed_difference=float(args.hdv_tm_speed_difference),
        hdv_tm_distance_to_leading_vehicle=float(args.hdv_tm_distance_to_leading_vehicle),
        hdv_tm_auto_lane_change=bool(args.hdv_tm_auto_lane_change),
        randomize_scenarios=bool(args.randomize_scenarios),
        relative_offset_jitter=float(args.relative_offset_jitter),
        randomize_lane_offsets=bool(args.randomize_lane_offsets),
        randomize_lane_offset_min=int(args.randomize_lane_offset_min),
        randomize_lane_offset_max=int(args.randomize_lane_offset_max),
        randomize_spawn_start_index=False,
        shuffle_spawn_points=False,
        low_speed_alpha=0.5,
        low_risk_threshold=0.25,
        front_slow_alpha=0.6,
        front_slow_gap=35.0,
        no_cuda=bool(args.no_cuda),
        torch_seed=int(args.torch_seed),
        min_num_adv=0,
        max_num_adv=0,
        purge_existing_actors_on_reset=bool(args.purge_existing_actors_on_reset),
        cleanup_destroy_mode=str(args.cleanup_destroy_mode),
    )


def make_natural_spec(args):
    if args.scenario_preset not in finetune.SCENARIO_PRESETS:
        raise ValueError(f"Unknown scenario preset: {args.scenario_preset}")
    rng = np.random.RandomState(0)
    spec = finetune.make_env_spec(
        env_id=0,
        template=args.scenario_preset,
        adv_step=0,
        num_adv=0,
        rng=rng,
    )
    spec.num_adv = 0
    spec.num_natural = int(args.num_natural)
    return spec


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def episode_row(checkpoint: Path, seed: int, metrics, train_mod, label_with_source: bool = False):
    ego_crash = bool(metrics.get("crash", False))
    row = {
        "model_group": "training_final" if "joint_" in str(checkpoint) else "finetuned",
        "model_label": checkpoint_label(checkpoint, label_with_source),
        "model_source": checkpoint_source(checkpoint),
        "checkpoint_step": checkpoint_step(checkpoint),
        "ego_checkpoint": str(checkpoint),
        "seed": int(seed),
        "ego_crash": ego_crash,
        "combined_efficiency": float(metrics.get("combined_efficiency", 0.0)),
        "traffic_efficiency": finetune.efficiency_from_metrics(metrics),
        "progress_auc": float(metrics.get("progress_auc", 0.0)),
        "route_speed_efficiency": finetune.route_speed_efficiency_from_metrics(metrics),
        "route_completion": metrics.get("route_completion", 0.0),
        "route_progress_m": metrics.get("route_progress_m", finetune.route_progress_from_mapping(metrics)),
        "ego_route_distance_m": metrics.get("ego_route_distance_m", metrics.get("route_progress_m", 0.0)),
        "elapsed_steps": metrics.get("steps", 0),
        "elapsed_time_s": float(metrics.get("elapsed_time_s", float(metrics.get("steps", 0)) * finetune.metrics_dt(metrics))),
        "fixed_delta_seconds": finetune.metrics_dt(metrics),
        "mean_speed": metrics.get("average_speed", 0.0),
        "max_speed": metrics.get("max_speed", metrics.get("average_speed", 0.0)),
        "min_speed": metrics.get("min_speed", metrics.get("average_speed", 0.0)),
        "low_speed_ratio": float(metrics.get("low_speed_ratio", 1.0)),
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
    }
    row.update(train_mod.episode_log_row(0, metrics, {}))
    row["ego_crash"] = ego_crash
    return row


def summarize_model_rows(rows):
    summary = {
        "model_label": rows[0]["model_label"],
        "checkpoint_step": int(rows[0]["checkpoint_step"]),
        "ego_checkpoint": rows[0]["ego_checkpoint"],
        "episodes": len(rows),
    }
    for key in METRIC_KEYS:
        values = []
        for row in rows:
            try:
                values.append(float(row.get(key, 0.0)))
            except (TypeError, ValueError):
                pass
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"min_{key}"] = float(np.min(values))
            summary[f"max_{key}"] = float(np.max(values))
    for key in RATE_KEYS:
        values = [float(bool(row.get(key, False))) for row in rows]
        summary[f"{key}_rate"] = float(np.mean(values)) if values else 0.0
    return summary


def write_comparison_tables(output_dir: Path, all_rows):
    df = pd.DataFrame(all_rows)
    metric_cols = [key for key in METRIC_KEYS if key in df.columns]
    pivot = df.pivot_table(
        index="seed",
        columns="model_label",
        values=metric_cols,
        aggfunc="mean",
    )
    pivot.to_csv(output_dir / "seed_metric_pivot.csv")
    seed_summary = df.groupby("seed", as_index=False).agg({key: "mean" for key in metric_cols})
    seed_summary.to_csv(output_dir / "seed_summary.csv", index=False)


def write_requested_metric_table(output_dir: Path, all_rows):
    df = pd.DataFrame(all_rows)
    if df.empty:
        return
    if "ego_crash" not in df.columns and "crash" in df.columns:
        df["ego_crash"] = df["crash"]
    rows = []
    for (model_label, checkpoint_step), sub in df.groupby(["model_label", "checkpoint_step"], dropna=False):
        row = {
            "model_label": model_label,
            "checkpoint_step": int(checkpoint_step),
            "episode_count": int(len(sub)),
        }
        if "ego_checkpoint" in sub.columns:
            row["ego_checkpoint"] = str(sub["ego_checkpoint"].iloc[0])
        for source_col, output_name in REQUESTED_TABLE_METRICS:
            if source_col not in sub.columns:
                continue
            values = pd.to_numeric(sub[source_col], errors="coerce").dropna()
            row[f"{output_name}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{output_name}_var"] = float(values.var(ddof=0)) if len(values) else np.nan
            row[f"{output_name}_mean_var"] = format_mean_var(
                row[f"{output_name}_mean"],
                row[f"{output_name}_var"],
            )
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("checkpoint_step")
    table.to_csv(output_dir / "requested_metrics_mean_variance.csv", index=False)


def scenario_family_from_manifest(output_dir: Path) -> str:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return "natural_traffic"
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "natural_traffic"
    env_config = manifest.get("env_config", {})
    return str(env_config.get("scenario_family") or manifest.get("scenario_preset") or "natural_traffic")


def write_scenario_family_metric_table(output_dir: Path, all_rows):
    df = pd.DataFrame(all_rows)
    if df.empty:
        return
    if "ego_crash" not in df.columns and "crash" in df.columns:
        df["ego_crash"] = df["crash"]
    if "model_group" not in df.columns:
        df["model_group"] = df.get("ego_checkpoint", "").map(
            lambda value: "training_final" if "joint_" in str(value) else "finetuned"
        )
    if "scenario_family" not in df.columns:
        df["scenario_family"] = scenario_family_from_manifest(output_dir)

    rows = []
    group_cols = ["model_group", "model_label", "checkpoint_step", "scenario_family"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["episodes"] = int(len(sub))
        for source_col, output_name in SCENARIO_FAMILY_TABLE_METRICS:
            if source_col not in sub.columns:
                continue
            values = pd.to_numeric(sub[source_col], errors="coerce").dropna()
            mean_value = float(values.mean()) if len(values) else np.nan
            var_value = float(values.var(ddof=0)) if len(values) else np.nan
            row[f"{output_name}_mean_var"] = format_mean_var(mean_value, var_value)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return
    table = table.sort_values(["scenario_family", "checkpoint_step"])
    table.to_csv(output_dir / "scenario_family_comparison_mean_plus_variance.csv", index=False)


def format_mean_var(mean_value, var_value) -> str:
    if pd.isna(mean_value):
        return ""
    return f"{float(mean_value):.4f} ± {float(var_value):.4f}"


def seed_mean_variance(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_label", "checkpoint_step", "seed"]
    rows = []
    metrics = [(key, key) for key in [*METRIC_KEYS, *RATE_KEYS] if key in df.columns]
    for keys, sub in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["episodes"] = int(len(sub))
        for metric, zh_name in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_var"] = float(values.var(ddof=0)) if len(values) else np.nan
            row[f"{zh_name}_mean_var"] = format_mean_var(row[f"{metric}_mean"], row[f"{metric}_var"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["seed", "checkpoint_step"])


def model_mean_variance(seed_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model_label", "checkpoint_step"]
    metric_names = sorted({
        col[:-5] for col in seed_table.columns
        if col.endswith("_mean") and f"{col[:-5]}_var" in seed_table.columns
    })
    for keys, sub in seed_table.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["scenario_count"] = int(len(sub))
        row["episode_count"] = int(sub["episodes"].sum()) if "episodes" in sub.columns else int(len(sub))
        for metric in metric_names:
            mean_col = f"{metric}_mean"
            var_col = f"{metric}_var"
            metric_mean = float(pd.to_numeric(sub[mean_col], errors="coerce").mean())
            metric_var = float(pd.to_numeric(sub[var_col], errors="coerce").mean())
            row[f"{metric}_mean"] = metric_mean
            row[f"{metric}_var"] = metric_var
            row[f"{metric}_mean_var"] = format_mean_var(metric_mean, metric_var)
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


def save_model_bar(plt, model_table: pd.DataFrame, metric: str, title: str, output_path: Path):
    mean_col = f"{metric}_mean"
    if mean_col not in model_table.columns:
        return False
    table = model_table.sort_values("checkpoint_step")
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(table)), 5.0))
    x = np.arange(len(table))
    ax.bar(x, table[mean_col], color="#4c78a8")
    ax.set_title(f"{title} comparison")
    ax.set_xlabel("Ego model")
    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels(table["model_label"], rotation=45, ha="right")
    for idx, value in enumerate(table[mean_col]):
        if pd.notna(value):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def save_seed_grouped_bar(plt, seed_table: pd.DataFrame, metric: str, title: str, output_path: Path):
    mean_col = f"{metric}_mean"
    if mean_col not in seed_table.columns:
        return False
    seeds = sorted(pd.to_numeric(seed_table["seed"], errors="coerce").dropna().astype(int).unique())
    if not seeds:
        return False
    model_order = (
        seed_table[["model_label", "checkpoint_step"]]
        .drop_duplicates()
        .sort_values("checkpoint_step")["model_label"]
        .tolist()
    )
    ncols = 2
    nrows = int(np.ceil(len(seeds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(4.0 * nrows, 5.5)), squeeze=False)
    for ax, seed in zip(axes.flatten(), seeds):
        sub = seed_table[seed_table["seed"].astype(int) == int(seed)].copy()
        sub["model_label"] = pd.Categorical(sub["model_label"], categories=model_order, ordered=True)
        sub = sub.sort_values("model_label")
        x = np.arange(len(sub))
        ax.bar(x, sub[mean_col], color="#4c78a8")
        ax.set_title(f"Seed-{seed}")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model_label"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(title)
        for idx, value in enumerate(sub[mean_col]):
            if pd.notna(value):
                ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=6)
    for ax in axes.flatten()[len(seeds):]:
        ax.axis("off")
    fig.suptitle(f"{title} under each natural-traffic seed", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def save_seed_compact_grouped_bar(plt, seed_table: pd.DataFrame, metric: str, title: str, output_path: Path):
    mean_col = f"{metric}_mean"
    if mean_col not in seed_table.columns:
        return False
    pivot = seed_table.pivot_table(index="seed", columns="model_label", values=mean_col, aggfunc="mean").sort_index()
    if pivot.empty:
        return False
    model_order = (
        seed_table[["model_label", "checkpoint_step"]]
        .drop_duplicates()
        .sort_values("checkpoint_step")["model_label"]
        .tolist()
    )
    pivot = pivot[[label for label in model_order if label in pivot.columns]]
    fig, ax = plt.subplots(figsize=(max(11, 0.7 * len(pivot.columns)), 5.5))
    x = np.arange(len(pivot.index))
    width = min(0.8 / max(len(pivot.columns), 1), 0.08)
    for idx, column in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2.0) * width
        ax.bar(x + offset, pivot[column], width=width, label=column)
    ax.set_title(f"{title} by natural-traffic seed")
    ax.set_xlabel("Scenario seed")
    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(item)) for item in pivot.index], rotation=30, ha="right")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def generate_tables_and_figures(output_dir: Path, all_rows):
    df = pd.DataFrame(all_rows)
    if df.empty:
        return []
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    seed_table = seed_mean_variance(df)
    model_table = model_mean_variance(seed_table)
    seed_table.to_csv(output_dir / "scenario_mean_variance.csv", index=False)
    model_table.to_csv(output_dir / "model_mean_variance_numeric.csv", index=False)
    model_table.to_csv(output_dir / "model_comparison_mean_plus_variance.csv", index=False)
    write_scenario_family_metric_table(output_dir, all_rows)

    plt = setup_matplotlib()
    figures = []
    for metric, title in PLOT_METRICS:
        model_path = figures_dir / f"{metric}_bar.png"
        if save_model_bar(plt, model_table, metric, title, model_path):
            figures.append(model_path)
        compact_path = figures_dir / f"scenario_mean_{metric}_bar.png"
        if save_seed_compact_grouped_bar(plt, seed_table, metric, title, compact_path):
            figures.append(compact_path)
        by_seed_path = figures_dir / f"by_seed_{metric}_bar.png"
        if save_seed_grouped_bar(plt, seed_table, metric, title, by_seed_path):
            figures.append(by_seed_path)
    return figures


def main():
    args = parse_args()
    output_dir = make_output_dir(args)
    selected = select_all_checkpoints(args)
    seeds = scenario_seeds(args)
    env_args = make_env_args(args)
    train_mod, _MAPPOAgent, EgoPPOAdapter, make_env = finetune.runtime_imports()

    if str(args.hdv_control_mode).lower() == "policy" and str(args.hdv_model).strip():
        hdv_path = finetune.latest_hdv_model(MODEL_ROOT / "results") if args.hdv_model == "latest" else Path(args.hdv_model)
        env_args._hdv_policy = train_mod.load_hdv_policy(str(hdv_path))
    else:
        hdv_path = None

    spec = make_natural_spec(args)
    env_config = finetune.env_config_from_spec(spec, env_args)
    env = make_env(env_config, config_path=args.config)

    manifest = {
        "ego_dir": str(args.ego_dir),
        "selected_checkpoints": [str(path) for path in selected],
        "selected_steps": [checkpoint_step(path) for path in selected],
        "scenario_seeds": seeds,
        "scenario_preset": args.scenario_preset,
        "num_natural": int(args.num_natural),
        "hdv_control_mode": str(args.hdv_control_mode),
        "hdv_model": str(hdv_path or ""),
        "env_config": env_config,
        "args": vars(args),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Output dir: {output_dir}", flush=True)
    print(f"Selected {len(selected)} checkpoints:", flush=True)
    print(", ".join(str(checkpoint_step(path)) for path in selected), flush=True)
    print(f"Scenario seeds: {', '.join(str(seed) for seed in seeds)}", flush=True)

    all_rows = []
    summary_rows = []
    episode_log_path = output_dir / "episode_log.csv"
    if args.skip_existing and episode_log_path.exists():
        all_rows = pd.read_csv(episode_log_path).to_dict("records")
    try:
        for checkpoint in selected:
            label = checkpoint_label(checkpoint, bool(args.label_with_source))
            if args.skip_existing and episode_log_path.exists():
                existing = pd.DataFrame(all_rows)
                done = set(existing.loc[existing["model_label"] == label, "seed"].astype(int).tolist())
                if set(seeds).issubset(done):
                    print(f"[SKIP] {label}", flush=True)
                    rows_for_model = existing[existing["model_label"] == label].to_dict("records")
                    if rows_for_model:
                        summary_rows.append(summarize_model_rows(rows_for_model))
                    continue
            ego = EgoPPOAdapter.from_config({"use_cuda": not args.no_cuda, "seed": args.torch_seed}, env.n_s, env.n_a)
            ego.load(str(checkpoint))
            rows_for_model = []
            for seed in seeds:
                metrics = finetune.run_eval_episode(env, None, ego, train_mod, env_args, int(seed))
                row = episode_row(checkpoint, int(seed), metrics, train_mod, bool(args.label_with_source))
                rows_for_model.append(row)
                all_rows.append(row)
                print(
                    f"{label} seed={seed} eff={row['combined_efficiency']:.3f} "
                    f"speed={float(row['mean_speed']):.2f} dist={float(row['ego_route_distance_m']):.1f} "
                    f"crash={row.get('crash', False)}",
                    flush=True,
                )
            summary_rows.append(summarize_model_rows(rows_for_model))
            write_csv(episode_log_path, all_rows)
            write_csv(output_dir / "overall_checkpoint_comparison.csv", summary_rows)
    finally:
        env.close()

    write_comparison_tables(output_dir, all_rows)
    write_requested_metric_table(output_dir, all_rows)
    figures = generate_tables_and_figures(output_dir, all_rows)
    print("=" * 80, flush=True)
    print(f"Episode log: {episode_log_path}", flush=True)
    print(f"Overall table: {output_dir / 'overall_checkpoint_comparison.csv'}", flush=True)
    print(f"Formatted table: {output_dir / 'model_comparison_mean_plus_variance.csv'}", flush=True)
    print(f"Requested metrics table: {output_dir / 'requested_metrics_mean_variance.csv'}", flush=True)
    print(f"Scenario table: {output_dir / 'scenario_mean_variance.csv'}", flush=True)
    print(f"Seed table: {output_dir / 'seed_summary.csv'}", flush=True)
    for figure in figures:
        print(f"Figure: {figure}", flush=True)


if __name__ == "__main__":
    main()
