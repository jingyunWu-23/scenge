"""Generate CARLA-Evolution-style metrics from SafeBench evaluation records.

The input is a SafeBench ``eval_results`` directory, an experiment directory
containing one, or a single ``*_records.pkl``/``records.pkl`` file. The output
uses the same core metric names and mean/variance convention as the migrated
``carla_evolution`` normal and frozen-MAPPO evaluations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from safebench.scenario.scenario_definition.atomic_criteria import Status
except Exception:  # pragma: no cover - allows reading plain dicts outside CARLA envs
    Status = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_TYPE = REPO_ROOT / "safebench/scenario/config/scenario_type/standard.json"

SCENARIO_NAMES = {
    1: "DynamicObjectCrossing",
    2: "VehicleTurningRoute",
    3: "OtherLeadingVehicle",
    4: "ManeuverOppositeDirection",
}

SCENARIO_ROUTE_LENGTHS = {
    1: 150.0,
    2: 180.0,
    3: 200.0,
    4: 200.0,
}

REQUESTED_METRICS = [
    "route_completion",
    "ego_route_distance_m",
    "mean_speed",
    "combined_efficiency",
    "episode_reward",
    "ego_crash",
    "progress_auc",
    "route_speed_efficiency",
    "low_speed_ratio",
]

FIGURE_METRICS = [
    ("route_completion", "Route Completion"),
    ("ego_route_distance_m", "Ego Route Distance (m)"),
    ("mean_speed", "Mean Speed"),
    ("combined_efficiency", "Combined Efficiency"),
    ("ego_crash", "Ego Crash Rate"),
    ("progress_auc", "Progress AUC"),
    ("route_speed_efficiency", "Route Speed Efficiency"),
]


def _status_failure(value: Any) -> bool:
    if Status is not None and value == Status.FAILURE:
        return True
    text = str(value).lower()
    return text.endswith("failure") or text == "failure" or value is True or value == 1


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _records_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    eval_dir = input_path / "eval_results"
    root = eval_dir if eval_dir.exists() else input_path
    files = sorted(root.glob("*_records.pkl"))
    plain = root / "records.pkl"
    if plain.exists():
        files.append(plain)
    return files


def _load_data_id_map(path: Path) -> dict[int, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {int(row["data_id"]): row for row in rows}


def _infer_model_label(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    parts = path.parts
    if "log" in parts:
        try:
            idx = parts.index("log")
            return "_".join(parts[idx + 2 : idx + 5])
        except Exception:
            pass
    return path.parent.parent.name if path.parent.name == "eval_results" else path.stem


def _route_length(sequence: list[dict[str, Any]], scenario_id: int) -> float:
    final = sequence[-1] if sequence else {}
    completion = _safe_float(final.get("route_complete")) / 100.0
    distance = max((_safe_float(item.get("driven_distance")) for item in sequence), default=0.0)
    if completion > 1e-6 and distance > 0.0:
        return max(distance / min(completion, 1.0), distance)
    return SCENARIO_ROUTE_LENGTHS.get(int(scenario_id), max(distance, 1.0))


def _progress_auc(sequence: list[dict[str, Any]], max_steps: int, completed: bool) -> float:
    progress_ratio = np.asarray(
        [np.clip(_safe_float(item.get("route_complete")) / 100.0, 0.0, 1.0) for item in sequence],
        dtype=np.float64,
    )
    if len(progress_ratio) > max_steps:
        progress_ratio = progress_ratio[:max_steps]
    if len(progress_ratio) < max_steps:
        padding_value = 1.0 if completed else (float(progress_ratio[-1]) if len(progress_ratio) else 0.0)
        progress_ratio = np.pad(
            progress_ratio,
            (0, max_steps - len(progress_ratio)),
            constant_values=padding_value,
        )
    return float(np.mean(progress_ratio)) if len(progress_ratio) else 0.0


def _episode_reward(sequence: list[dict[str, Any]]) -> float:
    rewards = [item.get("ego_reward") for item in sequence if "ego_reward" in item]
    if not rewards:
        return np.nan
    return float(np.sum([_safe_float(value) for value in rewards]))


def _episode_row(
    data_id: int,
    sequence: list[dict[str, Any]],
    source_file: Path,
    data_id_map: dict[int, dict[str, Any]],
    model_label: str,
    max_steps: int,
    low_speed_alpha: float,
    cruise_speed: float,
) -> dict[str, Any]:
    meta = data_id_map.get(int(data_id), {})
    scenario_id = int(meta.get("scenario_id", 0))
    route_id = int(meta.get("route_id", -1))
    scenario_name = SCENARIO_NAMES.get(scenario_id, f"scenario_{scenario_id}")
    final = sequence[-1] if sequence else {}

    route_length = _route_length(sequence, scenario_id)
    ego_route_distance_m = max((_safe_float(item.get("driven_distance")) for item in sequence), default=0.0)
    route_completion = np.clip(_safe_float(final.get("route_complete")) / 100.0, 0.0, 1.0)
    completed = route_completion >= 0.99
    elapsed_time_s = max(
        _safe_float(final.get("current_game_time")) - _safe_float(sequence[0].get("current_game_time")) if sequence else 0.0,
        0.0,
    )
    if elapsed_time_s <= 0.0:
        elapsed_time_s = float(len(sequence))

    speeds = np.asarray([_safe_float(item.get("ego_velocity")) for item in sequence], dtype=np.float64)
    mean_speed = float(np.mean(speeds)) if len(speeds) else 0.0
    low_speed_steps = int(np.sum(speeds < low_speed_alpha * cruise_speed)) if len(speeds) else 0
    low_risk_steps = len(sequence)
    low_speed_ratio = float(low_speed_steps) / max(float(low_risk_steps), 1.0)
    progress_auc = _progress_auc(sequence, max_steps, completed)
    ego_crash = any(_status_failure(item.get("collision")) for item in sequence)

    return {
        "model_label": model_label,
        "source_file": str(source_file),
        "data_id": int(data_id),
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "scenario_family": scenario_name,
        "route_id": route_id,
        "seed": int(data_id),
        "steps": int(len(sequence)),
        "elapsed_time_s": float(elapsed_time_s),
        "route_length": float(route_length),
        "route_completion": float(route_completion),
        "route_progress_m": float(route_completion * route_length),
        "ego_route_distance_m": float(ego_route_distance_m),
        "mean_speed": mean_speed,
        "max_speed": float(np.max(speeds)) if len(speeds) else 0.0,
        "min_speed": float(np.min(speeds)) if len(speeds) else 0.0,
        "progress_auc": progress_auc,
        "low_risk_steps": int(low_risk_steps),
        "low_speed_steps": int(low_speed_steps),
        "low_speed_ratio": low_speed_ratio,
        "combined_efficiency": float(0.8 * progress_auc + 0.2 * (1.0 - low_speed_ratio)),
        "route_speed_efficiency": float(ego_route_distance_m / max(elapsed_time_s, 1e-6)),
        "traffic_efficiency": float(0.8 * progress_auc + 0.2 * (1.0 - low_speed_ratio)),
        "episode_reward": _episode_reward(sequence),
        "crash": bool(ego_crash),
        "ego_crash": bool(ego_crash),
        "front_slow_flag": np.nan,
        "safe_lane_change_opportunity_flag": np.nan,
        "ego_lane_change_success_flag": np.nan,
        "ego_overtake_success_flag": np.nan,
    }


def _format_mean_var(mean_value: float, var_value: float) -> str:
    if pd.isna(mean_value):
        return ""
    return f"{float(mean_value):.4f} ± {float(var_value):.4f}"


def _mean_var_table(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["episodes"] = int(len(sub))
        for metric in metrics:
            if metric not in sub.columns:
                continue
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            mean_value = float(values.mean()) if len(values) else np.nan
            var_value = float(values.var(ddof=0)) if len(values) else np.nan
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_var"] = var_value
            row[f"{metric}_mean_var"] = _format_mean_var(mean_value, var_value)
        rows.append(row)
    return pd.DataFrame(rows)


def _save_figures(output_dir: Path, model_table: pd.DataFrame, scenario_table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    for metric, title in FIGURE_METRICS:
        mean_col = f"{metric}_mean"
        if mean_col not in model_table.columns:
            continue
        table = model_table.sort_values("model_label")
        fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * len(table)), 4.5))
        ax.bar(np.arange(len(table)), table[mean_col], color="#4c78a8")
        ax.set_xticks(np.arange(len(table)))
        ax.set_xticklabels(table["model_label"], rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(figures / f"model_{metric}.png")
        plt.close(fig)

    for metric, title in FIGURE_METRICS:
        mean_col = f"{metric}_mean"
        if mean_col not in scenario_table.columns:
            continue
        table = scenario_table.sort_values(["scenario_id", "route_id"])
        labels = table["scenario_name"].astype(str) + "-r" + table["route_id"].astype(str)
        fig, ax = plt.subplots(figsize=(max(10.0, 0.55 * len(table)), 4.8))
        ax.bar(np.arange(len(table)), table[mean_col], color="#59a14f")
        ax.set_xticks(np.arange(len(table)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"Scenario {title}")
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(figures / f"scenario_{metric}.png")
        plt.close(fig)


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_path = Path(args.input).expanduser().resolve()
    data_id_map = _load_data_id_map(Path(args.scenario_type).expanduser().resolve())
    rows = []
    for record_file in _records_files(input_path):
        records = joblib.load(record_file)
        if not isinstance(records, dict):
            continue
        model_label = _infer_model_label(record_file, args.model_label)
        for data_id, sequence in records.items():
            if not sequence:
                continue
            rows.append(
                _episode_row(
                    int(data_id),
                    sequence,
                    record_file,
                    data_id_map,
                    model_label,
                    args.max_steps,
                    args.low_speed_alpha,
                    args.cruise_speed,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="SafeBench eval directory or records pkl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario-type", default=str(DEFAULT_SCENARIO_TYPE))
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--low-speed-alpha", type=float, default=0.5)
    parser.add_argument("--cruise-speed", type=float, default=8.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args)
    if not rows:
        raise SystemExit("No SafeBench record rows found. Pass an eval_results directory or *_records.pkl file.")

    episode_df = pd.DataFrame(rows).sort_values(["model_label", "scenario_id", "route_id", "data_id"])
    episode_df.to_csv(output_dir / "episode_log.csv", index=False)

    scenario_table = _mean_var_table(
        episode_df,
        ["model_label", "scenario_id", "scenario_name", "route_id"],
        REQUESTED_METRICS,
    ).sort_values(["model_label", "scenario_id", "route_id"])
    scenario_table.to_csv(output_dir / "scenario_mean_variance.csv", index=False)

    model_table = _mean_var_table(
        episode_df,
        ["model_label"],
        REQUESTED_METRICS,
    ).sort_values("model_label")
    model_table.to_csv(output_dir / "model_comparison_mean_plus_variance.csv", index=False)
    model_table.to_csv(output_dir / "requested_metrics_mean_variance.csv", index=False)

    family_table = _mean_var_table(
        episode_df,
        ["model_label", "scenario_family"],
        REQUESTED_METRICS,
    ).sort_values(["scenario_family", "model_label"])
    family_table.to_csv(output_dir / "scenario_family_comparison_mean_plus_variance.csv", index=False)

    _save_figures(output_dir, model_table, scenario_table)
    print(f"Wrote CARLA-style SafeBench metrics to {output_dir}")


if __name__ == "__main__":
    main()
