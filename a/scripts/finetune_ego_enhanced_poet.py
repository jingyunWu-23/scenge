"""Enhanced-POET-inspired fine-tuning for conservative ego behavior.

The trainable policy is the current EgoPPO policy. Frozen historical EgoPPO
checkpoints are only used to score whether an evolved environment exposes
conservative behavior in the current policy.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = MODEL_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from carla_evolution.scripts.evaluate_ego_adv import SCENARIO_PRESETS


DEFAULT_ADV_STEPS = [1300, 3350, 6850, 9150, 10600, 11700, 13600, 15100, 15700, 17250]
DEFAULT_EGO_CSV = MODEL_ROOT / "md" / "主车冻结策略集_推荐20个.csv"
DEFAULT_CONFIG = MODEL_ROOT / "configs" / "carla_0915.yaml"
DEFAULT_RESULTS = MODEL_ROOT / "results"
DEFAULT_SELECTED_EGO_CSV = DEFAULT_RESULTS / "joint_Jul_01_14_59_59" / "selected_zero_crash_ego_reward_gt80_after_round30.csv"


@dataclass
class AdvCheckpoint:
    step: int
    model_dir: str


@dataclass
class EgoCheckpoint:
    label: str
    step: int
    checkpoint: str
    round_no: str = ""
    stage: str = ""


@dataclass
class EnvironmentSpec:
    environment_id: int
    template: str
    adv_step: int
    num_adv: int
    num_natural: int
    adv_slots: list
    offsets: list
    lane_offsets: list
    target_speed_cruise: float = 20.0
    route_length: float = 200.0
    parent_environment_id: int = -1
    mutation_type: str = "initial"
    generation_created: int = 0
    score: float = 0.0
    parent_count: int = 0
    near_count: int = 0


class CarlaEpisodeRestartRequested(RuntimeError):
    def __init__(self, episode, checkpoint_path, exit_code):
        super().__init__(f"CARLA restart requested at ego episode {episode}")
        self.episode = int(episode)
        self.checkpoint_path = Path(checkpoint_path)
        self.exit_code = int(exit_code)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune current EgoPPO with an Enhanced-POET-style conservative scenario population.")
    parser.add_argument("--backend", choices=["mock", "carla"], default="carla")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--base-dir", default=str(DEFAULT_RESULTS / "ego_enhanced_poet_finetune"))
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Resume an interrupted fine-tuning run from this output directory.",
    )
    parser.add_argument("--current-ego-checkpoint", default=None, help="Current trainable EgoPPO checkpoint. Defaults to latest checkpoint-N.pt under results/.")
    parser.add_argument("--frozen-ego-csv", default=str(DEFAULT_EGO_CSV))
    parser.add_argument(
        "--experiment-candidates-csv",
        default=str(DEFAULT_SELECTED_EGO_CSV),
        help="Candidate CSV with selected ego/adv checkpoints. Used when --frozen-ego-set-size > 0.",
    )
    parser.add_argument(
        "--frozen-ego-set-size",
        type=int,
        default=0,
        help="If >0, select this many frozen ego checkpoints from --experiment-candidates-csv instead of using --frozen-ego-csv directly.",
    )
    parser.add_argument(
        "--prefer-ego-round-after",
        type=int,
        default=60,
        help="When selecting a smaller frozen ego set, prefer candidate rows after this training round.",
    )
    parser.add_argument(
        "--disable-matched-adv-init",
        action="store_true",
        default=False,
        help="Do not derive initial adversarial checkpoints from selected candidate rows; use --adv-steps instead.",
    )
    parser.add_argument("--ego-search-dir", action="append", default=None, help="Search root for frozen/current ego checkpoints. Repeatable.")
    parser.add_argument("--adv-search-dir", action="append", default=None, help="Search root for MAPPO actor/critic checkpoints. Repeatable.")
    parser.add_argument("--adv-steps", default=",".join(str(item) for item in DEFAULT_ADV_STEPS))
    parser.add_argument("--hdv-model", default="latest", help="'latest', explicit SB3 .zip path, or empty string to disable natural-car policy model.")
    parser.add_argument("--hdv-action", default="keep_lane")
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--children-per-parent", type=int, default=2)
    parser.add_argument("--selection-episodes", type=int, default=5)
    parser.add_argument(
        "--train-episodes-per-generation",
        type=int,
        default=0,
        help="Total training episodes per generation. <=0 enables automatic population_size * train_episodes_per_env.",
    )
    parser.add_argument(
        "--train-episodes-per-env",
        type=int,
        default=10,
        help="Automatic per-generation training episodes assigned to each environment when --train-episodes-per-generation <= 0.",
    )
    parser.add_argument("--validation-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--total-non-ego", type=int, default=4)
    parser.add_argument("--min-num-adv", type=int, default=1)
    parser.add_argument("--max-num-adv", type=int, default=4)
    parser.add_argument("--efficiency-margin", type=float, default=0.02)
    parser.add_argument(
        "--better-seed-min-count",
        type=int,
        default=0,
        help="Minimum matched seeds where a frozen ego must beat the current ego to count as a parent signal. 0 disables this consistency gate.",
    )
    parser.add_argument(
        "--low-speed-alpha",
        type=float,
        default=0.5,
        help="Low-speed threshold as a fraction of cruise speed for the conservative-efficiency metric.",
    )
    parser.add_argument(
        "--low-risk-threshold",
        type=float,
        default=0.25,
        help="Only count low-speed steps when obstacle/lane risk is below this threshold.",
    )
    parser.add_argument(
        "--front-slow-alpha",
        type=float,
        default=0.6,
        help="Mark the front vehicle as slow when its speed is below this fraction of cruise speed.",
    )
    parser.add_argument(
        "--front-slow-gap",
        type=float,
        default=35.0,
        help="Maximum same-lane front gap in meters for marking a slow-front-vehicle condition.",
    )
    parser.add_argument(
        "--no-large-diff-patience",
        type=int,
        default=3,
        help="Stop if no frozen ego is at least efficiency-margin faster than the current ego for this many consecutive generations.",
    )
    parser.add_argument(
        "--no-improvement-patience",
        type=int,
        default=0,
        help="Stop if validation traffic efficiency does not improve for this many consecutive generations. <=0 disables this stop condition.",
    )
    parser.add_argument(
        "--improvement-min-delta",
        type=float,
        default=0.01,
        help="Minimum validation efficiency increase treated as an improvement.",
    )
    parser.add_argument("--scenario-presets", default="follow_straight,passing,lane_change,cut_in")
    parser.add_argument("--selection-seeds", default="101,102,103,104,105")
    parser.add_argument("--validation-seeds", default="1001,1002,1003,1004,1005,1006,1007,1008,1009,1010")
    parser.add_argument("--seed", type=int, default=669)
    parser.add_argument("--spawn-start-index", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--ego-batch-size", type=int, default=64)
    parser.add_argument("--ego-n-steps", type=int, default=2048)
    parser.add_argument("--ego-lr", type=float, default=1e-4)
    parser.add_argument("--torch-seed", type=int, default=669)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--carla-rpc-timeout", type=float, default=180.0)
    parser.add_argument(
        "--carla-restart-episode-interval",
        type=int,
        default=15000,
        help="Save checkpoint and restart CARLA every N cumulative ego episodes. <=0 disables this guard.",
    )
    parser.add_argument(
        "--carla-restart-command",
        default="",
        help="Optional shell command that restarts the local CARLA server. If empty, the script exits after saving a checkpoint so an outer loop or manual restart can resume from it.",
    )
    parser.add_argument(
        "--carla-restart-wait-seconds",
        type=float,
        default=30.0,
        help="Seconds to wait after --carla-restart-command before reconnecting to CARLA.",
    )
    parser.add_argument(
        "--carla-restart-exit-code",
        type=int,
        default=75,
        help="Exit code used after checkpointing when --carla-restart-command is empty.",
    )
    parser.add_argument(
        "--purge-existing-actors-on-reset",
        action="store_true",
        default=False,
        help="Before each CARLA reset, destroy existing vehicle.* actors and collision sensors in this server. Use only when this CARLA server is dedicated to this experiment.",
    )
    parser.add_argument(
        "--cleanup-destroy-mode",
        choices=["sequential", "batch"],
        default="sequential",
        help="Actor cleanup mode. sequential is slower per actor but avoids apply_batch_sync stalls on unstable CARLA servers.",
    )
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def checkpoint_steps(model_dir, pattern):
    regex = re.compile(pattern)
    steps = set()
    if not Path(model_dir).exists():
        return steps
    for path in Path(model_dir).iterdir():
        match = regex.fullmatch(path.name)
        if match:
            steps.add(int(match.group(1)))
    return steps


def has_mappo_step(model_dir, step):
    model_dir = Path(model_dir)
    return (model_dir / f"actor_{step}.pt").exists() and (model_dir / f"critic_{step}.pt").exists()


def resolve_adv_checkpoints(steps, search_dirs):
    roots = [Path(item) for item in search_dirs]
    resolved = []
    missing = []
    for step in steps:
        candidates = []
        for root in roots:
            if not root.exists():
                continue
            for actor in root.rglob(f"actor_{step}.pt"):
                model_dir = actor.parent
                if has_mappo_step(model_dir, step):
                    candidates.append(model_dir)
        if not candidates:
            missing.append(step)
            continue
        model_dir = max(candidates, key=lambda path: (path / f"actor_{step}.pt").stat().st_mtime)
        resolved.append(AdvCheckpoint(step=int(step), model_dir=str(model_dir)))
    if missing:
        raise FileNotFoundError(f"Missing MAPPO checkpoints for Adv steps: {missing}")
    return resolved


def latest_ego_checkpoint(search_dirs):
    candidates = []
    regex = re.compile(r"checkpoint-(\d+)\.pt$")
    for root in search_dirs:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("checkpoint-*.pt"):
            match = regex.fullmatch(path.name)
            if match:
                candidates.append((path.stat().st_mtime, int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError("No checkpoint-N.pt found for current ego.")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def latest_hdv_model(search_root):
    candidates = [path for path in Path(search_root).rglob("*.zip") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No HDV .zip model found under {search_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_ego_step(row):
    for key in ("ego_step", "累计Ego Episode", "Checkpoint标识", "checkpoint", "Checkpoint", "ego_checkpoint"):
        value = row.get(key)
        if value:
            match = re.search(r"(\d+)", str(value))
            if match:
                return int(match.group(1))
    raise ValueError(f"Cannot parse ego checkpoint step from row: {row}")


def resolve_checkpoint_by_step(step, search_dirs):
    name = f"checkpoint-{step}.pt"
    candidates = []
    for root in search_dirs:
        root = Path(root)
        if root.exists():
            candidates.extend(path for path in root.rglob(name) if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"Cannot find {name}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def relocate_stale_project_path(path_value):
    """Relocate paths saved from another machine into the current project tree."""
    path = Path(str(path_value or ""))
    if path.exists():
        return path
    parts = path.parts
    if "results" in parts:
        idx = parts.index("results")
        candidate = MODEL_ROOT.joinpath(*parts[idx:])
        if candidate.exists():
            return candidate
    return path


def resolve_adv_model_dir_from_candidate(row, requested_step, search_dirs):
    model_dir = relocate_stale_project_path(row.get("adv_model_dir") or "")
    if has_mappo_step(model_dir, requested_step) or available_mappo_steps(model_dir):
        return model_dir

    actor_name = f"actor_{int(requested_step)}.pt"
    candidates = []
    for root in search_dirs:
        root = Path(root)
        if not root.exists():
            continue
        for actor in root.rglob(actor_name):
            candidate_dir = actor.parent
            if has_mappo_step(candidate_dir, requested_step):
                candidates.append(candidate_dir)
    if candidates:
        return max(candidates, key=lambda path: (path / actor_name).stat().st_mtime)
    return model_dir


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def read_candidate_rows(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if as_int(row.get("ego_step")) > 0]
    rows.sort(key=lambda row: (as_int(row.get("round")), as_int(row.get("ego_step"))))
    if not rows:
        raise ValueError(f"No candidate ego rows found in {csv_path}")
    return rows


def uniform_sample_rows(rows, count):
    rows = list(rows)
    count = int(count)
    if count <= 0:
        return []
    if count >= len(rows):
        return rows
    indices = np.linspace(0, len(rows) - 1, count).astype(int).tolist()
    return [rows[index] for index in indices]


def select_candidate_rows(rows, count, prefer_after_round):
    rows = sorted(rows, key=lambda row: (as_int(row.get("round")), as_int(row.get("ego_step"))))
    count = int(count)
    if count <= 0 or count >= len(rows):
        return rows

    preferred = [row for row in rows if as_int(row.get("round")) > int(prefer_after_round)]
    early = [row for row in rows if as_int(row.get("round")) <= int(prefer_after_round)]
    if len(preferred) >= count:
        return uniform_sample_rows(preferred, count)

    selected = list(preferred)
    selected.extend(uniform_sample_rows(early, count - len(selected)))
    return sorted(selected, key=lambda row: (as_int(row.get("round")), as_int(row.get("ego_step"))))


def candidate_row_key(row):
    return as_int(row.get("round")), as_int(row.get("ego_step"))


def selected_rows_for_adv_init(all_rows, selected_ego_rows, population_size):
    population_size = int(population_size)
    if population_size <= 0:
        return []
    selected_ego_rows = sorted(selected_ego_rows, key=candidate_row_key)
    if population_size <= len(selected_ego_rows):
        return uniform_sample_rows(selected_ego_rows, population_size)

    selected = list(selected_ego_rows)
    selected_keys = {candidate_row_key(row) for row in selected}
    remaining = [row for row in all_rows if candidate_row_key(row) not in selected_keys]
    selected.extend(uniform_sample_rows(remaining, population_size - len(selected)))
    if len(selected) < population_size:
        selected.extend(uniform_sample_rows(all_rows, population_size - len(selected)))
    return selected[:population_size]


def available_mappo_steps(model_dir):
    model_dir = Path(model_dir)
    regex = re.compile(r"actor_(\d+)\.pt$")
    steps = []
    if not model_dir.exists():
        return steps
    for actor in model_dir.glob("actor_*.pt"):
        match = regex.fullmatch(actor.name)
        if not match:
            continue
        step = int(match.group(1))
        if has_mappo_step(model_dir, step):
            steps.append(step)
    return sorted(set(steps))


def nearest_available_mappo_step(model_dir, requested_step):
    requested_step = int(requested_step)
    if has_mappo_step(model_dir, requested_step):
        return requested_step, True
    steps = available_mappo_steps(model_dir)
    if not steps:
        raise FileNotFoundError(f"No MAPPO actor/critic checkpoints found under {model_dir}")
    return min(steps, key=lambda step: (abs(step - requested_step), step)), False


def ego_checkpoint_from_candidate_row(row, search_dirs):
    step = as_int(row.get("ego_step"))
    checkpoint = Path(row.get("ego_checkpoint") or "")
    if not checkpoint.is_file():
        checkpoint = resolve_checkpoint_by_step(step, search_dirs)
    return EgoCheckpoint(
        label=row.get("ego_model") or f"Ego-{step}",
        step=step,
        round_no=str(row.get("round", "")),
        stage="selected_zero_crash_reward_gt80",
        checkpoint=str(checkpoint),
    )


def select_experiment_assets(args, search_dirs, adv_search_dirs):
    candidate_rows = read_candidate_rows(args.experiment_candidates_csv)
    selected_ego_rows = select_candidate_rows(
        candidate_rows,
        int(args.frozen_ego_set_size),
        int(args.prefer_ego_round_after),
    )
    frozen_egos = [ego_checkpoint_from_candidate_row(row, search_dirs) for row in selected_ego_rows]

    adv_rows = []
    adv_steps = parse_int_list(args.adv_steps)
    if not args.disable_matched_adv_init:
        matched_rows = selected_rows_for_adv_init(candidate_rows, selected_ego_rows, int(args.population_size))
        adv_steps = []
        for row in matched_rows:
            requested_step = as_int(row.get("adv_step"))
            model_dir = resolve_adv_model_dir_from_candidate(row, requested_step, adv_search_dirs)
            resolved_step, exact = nearest_available_mappo_step(model_dir, requested_step)
            adv_steps.append(int(resolved_step))
            adv_rows.append({
                "round": as_int(row.get("round")),
                "ego_step": as_int(row.get("ego_step")),
                "ego_model": row.get("ego_model") or f"Ego-{as_int(row.get('ego_step'))}",
                "requested_adv_step": requested_step,
                "resolved_adv_step": int(resolved_step),
                "exact_adv_checkpoint": bool(exact),
                "adv_model_dir": str(model_dir),
                "ego_reward": as_float(row.get("ego_reward")),
                "ego_crash_rate": as_float(row.get("ego_crash_rate")),
                "adv_reward": as_float(row.get("adv_reward")),
                "adv_crash_rate": as_float(row.get("adv_crash_rate")),
            })

    return frozen_egos, adv_steps, selected_ego_rows, adv_rows


def read_frozen_ego_set(csv_path, search_dirs):
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    result = []
    for row in rows:
        step = parse_ego_step(row)
        result.append(EgoCheckpoint(
            label=row.get("Checkpoint标识") or f"Ego-{step}",
            step=step,
            round_no=row.get("轮次", ""),
            stage=row.get("训练阶段", ""),
            checkpoint=str(resolve_checkpoint_by_step(step, search_dirs)),
        ))
    if not result:
        raise ValueError(f"No frozen ego checkpoints found in {csv_path}")
    return result


def base_positions(template):
    preset = SCENARIO_PRESETS[template]
    offsets = list(preset["adv_offsets"])
    lanes = list(preset["adv_lane_offsets"])
    while len(offsets) < 4:
        offsets.append(75.0 + 20.0 * (len(offsets) - 3))
        lanes.append(len(offsets) % 3)
    return offsets[:4], lanes[:4]


def make_env_spec(env_id, template, adv_step, num_adv, rng, generation=0, parent=-1, mutation="initial"):
    offsets, lanes = base_positions(template)
    num_adv = int(num_adv)
    slots = list(range(4))
    rng.shuffle(slots)
    adv_slots = sorted(slots[:num_adv])
    return EnvironmentSpec(
        environment_id=int(env_id),
        template=template,
        adv_step=int(adv_step),
        num_adv=num_adv,
        num_natural=max(0, 4 - num_adv),
        adv_slots=adv_slots,
        offsets=[float(item) for item in offsets],
        lane_offsets=[int(item) for item in lanes],
        target_speed_cruise=float(SCENARIO_PRESETS[template].get("target_speed_cruise", 20.0)),
        route_length=float(SCENARIO_PRESETS[template].get("route_length", 200.0)),
        parent_environment_id=int(parent),
        mutation_type=mutation,
        generation_created=int(generation),
    )


def initialize_population(size, templates, adv_steps, args, rng):
    population = []
    adv_counts = list(range(int(args.min_num_adv), int(args.max_num_adv) + 1))
    for env_id in range(int(size)):
        template = templates[env_id % len(templates)]
        adv_step = adv_steps[env_id % len(adv_steps)]
        num_adv = adv_counts[env_id % len(adv_counts)]
        population.append(make_env_spec(env_id, template, adv_step, num_adv, rng))
    return population


def mutate_environment(parent, new_id, generation, templates, adv_steps, rng):
    child = copy.deepcopy(parent)
    child.environment_id = int(new_id)
    child.parent_environment_id = int(parent.environment_id)
    child.generation_created = int(generation)
    child.parent_count = 0
    child.near_count = 0
    child.score = 0.0
    mutation_choices = ["offset", "lane", "adv_ratio", "adv_checkpoint", "template", "target_speed"]
    mutation = mutation_choices[int(rng.randint(0, len(mutation_choices)))]
    child.mutation_type = mutation
    if mutation == "offset":
        idx = int(rng.randint(0, len(child.offsets)))
        child.offsets[idx] = float(np.clip(child.offsets[idx] + rng.uniform(-5.0, 5.0), -50.0, 90.0))
    elif mutation == "lane":
        idx = int(rng.randint(0, len(child.lane_offsets)))
        child.lane_offsets[idx] = int(np.clip(child.lane_offsets[idx] + rng.choice([-1, 1]), 0, 2))
    elif mutation == "adv_ratio":
        child.num_adv = int(np.clip(child.num_adv + rng.choice([-1, 1]), 1, 4))
        child.num_natural = 4 - child.num_adv
        slots = list(range(4))
        rng.shuffle(slots)
        child.adv_slots = sorted(slots[:child.num_adv])
    elif mutation == "adv_checkpoint":
        current = adv_steps.index(child.adv_step) if child.adv_step in adv_steps else 0
        next_index = int(np.clip(current + rng.choice([-1, 1]), 0, len(adv_steps) - 1))
        child.adv_step = int(adv_steps[next_index])
    elif mutation == "template":
        child.template = templates[int(rng.randint(0, len(templates)))]
    elif mutation == "target_speed":
        child.target_speed_cruise = float(np.clip(child.target_speed_cruise + rng.uniform(-1.0, 1.0), 16.0, 22.0))
    return child


def env_config_from_spec(spec, args):
    adv_offsets = [spec.offsets[i] for i in spec.adv_slots]
    adv_lanes = [spec.lane_offsets[i] for i in spec.adv_slots]
    hdv_slots = [i for i in range(4) if i not in spec.adv_slots]
    hdv_offsets = [spec.offsets[i] for i in hdv_slots]
    hdv_lanes = [spec.lane_offsets[i] for i in hdv_slots]
    config = {
        "backend": args.backend,
        "max_episode_steps": int(args.max_steps),
        "num_cav": int(spec.num_adv),
        "num_hdv": int(spec.num_natural),
        "num_background": 0,
        "timeout": float(args.carla_rpc_timeout),
        "randomize_scenarios": bool(getattr(args, "randomize_scenarios", False)),
        "spawn_start_index": int(args.spawn_start_index),
        "randomize_spawn_start_index": bool(getattr(args, "randomize_spawn_start_index", False)),
        "shuffle_spawn_points": bool(getattr(args, "shuffle_spawn_points", False)),
        "randomize_lane_offsets": bool(getattr(args, "randomize_lane_offsets", False)),
        "randomize_lane_offset_min": int(getattr(args, "randomize_lane_offset_min", 0)),
        "randomize_lane_offset_max": int(getattr(args, "randomize_lane_offset_max", 2)),
        "purge_existing_actors_on_reset": bool(getattr(args, "purge_existing_actors_on_reset", False)),
        "cleanup_destroy_mode": str(getattr(args, "cleanup_destroy_mode", "sequential")),
        "hdv_control_mode": str(getattr(args, "hdv_control_mode", "")),
        "traffic_manager_port": int(getattr(args, "traffic_manager_port", 8000)),
        "hdv_tm_speed_difference": float(getattr(args, "hdv_tm_speed_difference", 0.0)),
        "hdv_tm_distance_to_leading_vehicle": float(
            getattr(args, "hdv_tm_distance_to_leading_vehicle", 8.0)
        ),
        "hdv_tm_auto_lane_change": bool(getattr(args, "hdv_tm_auto_lane_change", True)),
        "route_name": f"poet_{spec.template}_{spec.environment_id}",
        "scenario_family": str(SCENARIO_PRESETS[spec.template].get("scenario_family", spec.template)),
        "behavior_type": str(SCENARIO_PRESETS[spec.template].get("behavior_type", "")),
        "route_length": float(spec.route_length),
        "ego_offset": 0.0,
        "ego_lane_offset": 1,
        "adv_offsets": adv_offsets,
        "adv_lane_offsets": adv_lanes,
        "hdv_offsets": hdv_offsets,
        "hdv_lane_offsets": hdv_lanes,
        "target_speed_cruise": float(spec.target_speed_cruise),
        "relative_offset_jitter": float(getattr(args, "relative_offset_jitter", 0.0)),
    }
    if args.render:
        config.update({"no_rendering_mode": False, "spectator_follow": True})
    return config


def make_output_dir(args):
    if args.resume_dir:
        output_dir = Path(args.resume_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"--resume-dir does not exist: {output_dir}")
        (output_dir / "models" / "ego").mkdir(parents=True, exist_ok=True)
        return output_dir
    stamp = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = Path(args.base_dir) / f"poet_finetune_{stamp}"
    (output_dir / "models" / "ego").mkdir(parents=True, exist_ok=True)
    return output_dir


def write_csv(path, rows, mode="w"):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    exists = path.exists() and mode == "a"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def metrics_dt(metrics, default=0.05):
    return float(metrics.get("fixed_delta_seconds", metrics.get("dt", default)))


def route_progress_from_mapping(data):
    route_s = data.get("ego_route_s", None)
    if route_s is not None:
        try:
            return max(0.0, float(route_s))
        except (TypeError, ValueError):
            pass
    try:
        completion_full = float(data.get("route_completion_full", 0.0))
        route_length = float(data.get("route_length", 0.0))
        return max(0.0, completion_full * route_length)
    except (TypeError, ValueError):
        return 0.0


def route_speed_efficiency_from_metrics(metrics):
    progress = route_progress_from_mapping(metrics)
    if progress <= 0.0:
        progress = float(metrics.get("route_completion", 0.0)) * float(metrics.get("route_length", 0.0))
    elapsed = float(metrics.get("steps", 0)) * metrics_dt(metrics)
    return progress / max(elapsed, 1e-6)


def efficiency_from_metrics(metrics):
    if "combined_efficiency" in metrics:
        return float(metrics.get("combined_efficiency", 0.0))
    return route_speed_efficiency_from_metrics(metrics)


def calculate_progress_auc(progress_history, route_length, max_steps, completed):
    route_length = float(route_length)
    if route_length <= 0.0:
        return 0.0
    progress_ratio = np.clip(np.asarray(progress_history, dtype=np.float64) / route_length, 0.0, 1.0)
    if len(progress_ratio) > int(max_steps):
        progress_ratio = progress_ratio[: int(max_steps)]
    if len(progress_ratio) < int(max_steps):
        if bool(completed):
            padding_value = 1.0
        elif len(progress_ratio) > 0:
            padding_value = float(progress_ratio[-1])
        else:
            padding_value = 0.0
        progress_ratio = np.pad(
            progress_ratio,
            (0, int(max_steps) - len(progress_ratio)),
            constant_values=padding_value,
        )
    return float(np.mean(progress_ratio))


def init_behavior_markers():
    return {
        "front_slow_steps": 0,
        "safe_lane_change_opportunity_steps": 0,
        "ego_lane_change_action_steps": 0,
        "ego_lane_change_success_steps": 0,
        "ego_overtake_success_steps": 0,
        "ego_lane_change_intent_steps": 0,
    }


def update_behavior_markers(markers, info, ego_action, ego_speed, cruise_speed, args):
    front_gap = float(info.get("ego_front_gap_control", info.get("ego_front_gap", -1.0)))
    front_speed = float(info.get("ego_front_speed_control", -1.0))
    target_lane_risk = float(info.get("ego_target_lane_obstacle_risk", 0.0))
    obstacle_risk = float(info.get("ego_obstacle_risk", 0.0))
    obstacle_risk_control = float(info.get("ego_obstacle_risk_control", 0.0))
    risk = max(obstacle_risk, obstacle_risk_control, target_lane_risk)
    safety_override = (
        bool(info.get("ego_obstacle_takeover_active", False))
        or bool(info.get("ego_obstacle_stop_active", False))
        or float(info.get("ego_safety_brake", 0.0)) > 1e-6
    )
    front_slow = (
        0.0 <= front_gap <= float(args.front_slow_gap)
        and 0.0 <= front_speed < float(args.front_slow_alpha) * float(cruise_speed)
    )
    safe_lane_change_opportunity = (
        bool(info.get("ego_obstacle_escape_available", False))
        and risk < float(args.low_risk_threshold)
        and target_lane_risk < float(args.low_risk_threshold)
        and not safety_override
    )
    try:
        action_value = int(ego_action)
    except (TypeError, ValueError):
        action_value = -1
    lane_change_action = action_value in (0, 2)
    lane_change_success = bool(info.get("ego_lane_change_success", False))
    overtake_success = bool(lane_change_success and front_slow and float(ego_speed) > front_speed)
    lane_change_intent = bool(str(info.get("ego_lane_change_intent", "")).strip())

    markers["front_slow_steps"] += int(front_slow)
    markers["safe_lane_change_opportunity_steps"] += int(safe_lane_change_opportunity)
    markers["ego_lane_change_action_steps"] += int(lane_change_action)
    markers["ego_lane_change_success_steps"] += int(lane_change_success)
    markers["ego_overtake_success_steps"] += int(overtake_success)
    markers["ego_lane_change_intent_steps"] += int(lane_change_intent)


def behavior_marker_metrics(markers, steps):
    steps = max(int(steps), 1)
    return {
        "front_slow_steps": int(markers["front_slow_steps"]),
        "front_slow_ratio": float(markers["front_slow_steps"]) / float(steps),
        "front_slow_flag": bool(markers["front_slow_steps"] > 0),
        "safe_lane_change_opportunity_steps": int(markers["safe_lane_change_opportunity_steps"]),
        "safe_lane_change_opportunity_ratio": float(markers["safe_lane_change_opportunity_steps"]) / float(steps),
        "safe_lane_change_opportunity_flag": bool(markers["safe_lane_change_opportunity_steps"] > 0),
        "ego_lane_change_action_steps": int(markers["ego_lane_change_action_steps"]),
        "ego_lane_change_action_ratio": float(markers["ego_lane_change_action_steps"]) / float(steps),
        "ego_lane_change_action_flag": bool(markers["ego_lane_change_action_steps"] > 0),
        "ego_lane_change_success_steps": int(markers["ego_lane_change_success_steps"]),
        "ego_lane_change_success_flag": bool(markers["ego_lane_change_success_steps"] > 0),
        "ego_overtake_success_steps": int(markers["ego_overtake_success_steps"]),
        "ego_overtake_success_flag": bool(markers["ego_overtake_success_steps"] > 0),
        "ego_lane_change_intent_steps": int(markers["ego_lane_change_intent_steps"]),
        "ego_lane_change_intent_flag": bool(markers["ego_lane_change_intent_steps"] > 0),
    }


def safe_from_records(records):
    if not records:
        return False
    return not any(bool(item.get("crash", False)) or bool(item.get("adv_ego_collision", False)) for item in records)


def summarize_records(records):
    if not records:
        return {
            "all_safe": False,
            "mean_efficiency": 0.0,
            "mean_progress_auc": 0.0,
            "mean_low_speed_ratio": 1.0,
            "mean_front_slow_ratio": 0.0,
            "front_slow_episode_rate": 0.0,
            "mean_safe_lane_change_opportunity_ratio": 0.0,
            "safe_lane_change_opportunity_episode_rate": 0.0,
            "mean_ego_lane_change_action_ratio": 0.0,
            "ego_lane_change_episode_rate": 0.0,
            "ego_lane_change_success_episode_rate": 0.0,
            "ego_overtake_success_episode_rate": 0.0,
            "crash_rate": 1.0,
            "route_completion": 0.0,
            "route_distance_m": 0.0,
            "avg_speed": 0.0,
            "max_speed": 0.0,
            "min_speed": 0.0,
        }
    efficiencies = [efficiency_from_metrics(item) for item in records]
    return {
        "all_safe": safe_from_records(records),
        "mean_efficiency": float(np.mean(efficiencies)),
        "mean_progress_auc": float(np.mean([float(item.get("progress_auc", 0.0)) for item in records])),
        "mean_low_speed_ratio": float(np.mean([float(item.get("low_speed_ratio", 1.0)) for item in records])),
        "mean_front_slow_ratio": float(np.mean([float(item.get("front_slow_ratio", 0.0)) for item in records])),
        "front_slow_episode_rate": float(np.mean([float(item.get("front_slow_flag", False)) for item in records])),
        "mean_safe_lane_change_opportunity_ratio": float(np.mean([float(item.get("safe_lane_change_opportunity_ratio", 0.0)) for item in records])),
        "safe_lane_change_opportunity_episode_rate": float(np.mean([float(item.get("safe_lane_change_opportunity_flag", False)) for item in records])),
        "mean_ego_lane_change_action_ratio": float(np.mean([float(item.get("ego_lane_change_action_ratio", 0.0)) for item in records])),
        "ego_lane_change_episode_rate": float(np.mean([float(item.get("ego_lane_change_action_flag", False)) for item in records])),
        "ego_lane_change_success_episode_rate": float(np.mean([float(item.get("ego_lane_change_success_flag", False)) for item in records])),
        "ego_overtake_success_episode_rate": float(np.mean([float(item.get("ego_overtake_success_flag", False)) for item in records])),
        "crash_rate": float(np.mean([float(item.get("crash", False)) for item in records])),
        "route_completion": float(np.mean([float(item.get("route_completion", 0.0)) for item in records])),
        "route_distance_m": float(np.mean([float(item.get("ego_route_distance_m", item.get("route_progress_m", 0.0))) for item in records])),
        "avg_speed": float(np.mean([float(item.get("average_speed", 0.0)) for item in records])),
        "max_speed": float(np.max([float(item.get("max_speed", item.get("average_speed", 0.0))) for item in records])),
        "min_speed": float(np.min([float(item.get("min_speed", item.get("average_speed", 0.0))) for item in records])),
    }


def paired_efficiency_gap(current_rows, frozen_rows):
    current_by_seed = {int(row["seed"]): float(row["combined_efficiency"]) for row in current_rows}
    gaps = []
    for row in frozen_rows:
        seed = int(row["seed"])
        if seed not in current_by_seed:
            continue
        current_score = current_by_seed[seed]
        frozen_score = float(row["combined_efficiency"])
        gaps.append((frozen_score - current_score) / max(abs(current_score), 1e-6))
    if not gaps:
        return 0.0, 0
    return float(np.mean(gaps)), int(np.sum(np.asarray(gaps, dtype=np.float64) > 0.0))


def load_hdv(args, train_mod):
    if not str(args.hdv_model).strip():
        return None
    hdv_path = latest_hdv_model(DEFAULT_RESULTS) if args.hdv_model == "latest" else Path(args.hdv_model)
    return train_mod.load_hdv_policy(str(hdv_path)), str(hdv_path)


def evaluate_policy_on_env(spec, ego_checkpoint, policy_type, policy_label, adv_lookup, args, seeds, train_mod, MAPPOAgent, EgoPPOAdapter, make_env):
    adv_ckpt = adv_lookup[int(spec.adv_step)]
    env = make_env(env_config_from_spec(spec, args), config_path=args.config)
    records = []
    try:
        mappo = MAPPOAgent.from_config({"reward_type": "agents_rewards", "use_cuda": not args.no_cuda, "torch_seed": args.torch_seed}, env.n_s, env.n_a)
        mappo.load(adv_ckpt.model_dir, adv_ckpt.step, train_mode=False)
        ego = EgoPPOAdapter.from_config({"use_cuda": not args.no_cuda, "seed": args.torch_seed}, env.n_s, env.n_a)
        ego.load(str(ego_checkpoint))
        for seed in seeds:
            metrics = run_eval_episode(env, mappo, ego, train_mod, args, int(seed))
            records.append(metrics)
    finally:
        env.close()
    summary = summarize_records(records)
    rows = []
    for seed, metrics in zip(seeds, records):
        rows.append({
            "environment_id": spec.environment_id,
            "policy_type": policy_type,
            "policy_checkpoint": str(ego_checkpoint),
            "policy_label": policy_label,
            "seed": int(seed),
            "traffic_efficiency": efficiency_from_metrics(metrics),
            "combined_efficiency": float(metrics.get("combined_efficiency", efficiency_from_metrics(metrics))),
            "progress_auc": float(metrics.get("progress_auc", 0.0)),
            "route_speed_efficiency": route_speed_efficiency_from_metrics(metrics),
            "original_reward": metrics.get("episode_reward", 0.0),
            "collision": metrics.get("crash", False),
            "route_completion": metrics.get("route_completion", 0.0),
            "route_progress_m": metrics.get("route_progress_m", route_progress_from_mapping(metrics)),
            "ego_route_distance_m": metrics.get("ego_route_distance_m", metrics.get("route_progress_m", route_progress_from_mapping(metrics))),
            "completion_horizon_m": metrics.get("route_completion_distance", 0.0),
            "elapsed_steps": metrics.get("steps", 0),
            "elapsed_time_s": float(metrics.get("elapsed_time_s", float(metrics.get("steps", 0)) * metrics_dt(metrics))),
            "fixed_delta_seconds": metrics_dt(metrics),
            "mean_speed": metrics.get("average_speed", 0.0),
            "max_speed": metrics.get("max_speed", metrics.get("average_speed", 0.0)),
            "min_speed": metrics.get("min_speed", metrics.get("average_speed", 0.0)),
            "low_speed_ratio": float(metrics.get("low_speed_ratio", 1.0)),
            "low_risk_steps": int(metrics.get("low_risk_steps", 0)),
            "low_speed_steps": int(metrics.get("low_speed_steps", 0)),
            "lane_change_cancel_count": metrics.get("ego_lane_change_cancelled_for_obstacle_steps", 0),
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
            "unused_safe_lane_time": "",
            "risk_recovery_time": "",
        })
    return summary, rows


def run_eval_episode(env, mappo, ego, train_mod, args, seed):
    state, reset_info = env.reset(seed=seed)
    n_agents = len(env.controlled_vehicles)
    ego_state = np.asarray(env.obs2, dtype=np.float32).flatten()
    episode_reward = 0.0
    last_info = {"route_completion": reset_info.get("route_completion", 0.0)}
    diagnostics = train_mod.init_episode_diagnostics()
    progress_history = []
    ego_speed_history = []
    low_risk_steps = 0
    low_speed_steps = 0
    behavior_markers = init_behavior_markers()
    cruise_speed = float(env.config.get("target_speed_cruise", 20.0))
    for step in range(int(args.max_steps)):
        if n_agents > 0:
            if mappo is None:
                raise RuntimeError("MAPPO policy is required when evaluating with adversarial CAVs.")
            adv_actions = mappo.action(state, n_agents)
        else:
            adv_actions = []
        ego_action, _, _ = ego.select_action(ego_state, deterministic=True)
        full_action = train_mod.append_hdv_actions(env, list(adv_actions) + [ego_action], args)
        next_state, global_reward, terminated, truncated, info = env.step(full_action)
        ego_reward = float(info.get("ego_reward", global_reward))
        episode_reward += ego_reward
        train_mod.update_episode_diagnostics(diagnostics, info)
        last_info = info
        progress_history.append(route_progress_from_mapping(info))
        ego_speed = float(getattr(getattr(env, "ego_vehicle", None), "speed", info.get("average_speed", 0.0)))
        ego_speed_history.append(ego_speed)
        risk = max(
            float(info.get("ego_obstacle_risk", 0.0)),
            float(info.get("ego_obstacle_risk_control", 0.0)),
            float(info.get("ego_target_lane_obstacle_risk", 0.0)),
        )
        safety_override = (
            bool(info.get("ego_obstacle_takeover_active", False))
            or bool(info.get("ego_obstacle_stop_active", False))
            or float(info.get("ego_safety_brake", 0.0)) > 1e-6
        )
        if risk < float(args.low_risk_threshold) and not safety_override:
            low_risk_steps += 1
            if ego_speed < float(args.low_speed_alpha) * cruise_speed:
                low_speed_steps += 1
        update_behavior_markers(behavior_markers, info, ego_action, ego_speed, cruise_speed, args)
        state = next_state
        ego_state = np.asarray(env.obs2, dtype=np.float32).flatten()
        if terminated or truncated:
            break
    rewards = np.asarray(last_info.get("agents_rewards", [0.0]), dtype=np.float32)
    metrics = train_mod.episode_metrics(step + 1, episode_reward, rewards, train_mod.merge_episode_diagnostics(last_info, diagnostics))
    completed = bool(last_info.get("success", False)) or float(metrics.get("route_completion", 0.0)) >= 0.99
    progress_auc = calculate_progress_auc(
        progress_history,
        float(metrics.get("route_length", env.config.get("route_length", 200.0))),
        int(args.max_steps),
        completed,
    )
    low_speed_ratio = float(low_speed_steps) / max(float(low_risk_steps), 1.0)
    fallback_speed = float(metrics.get("average_speed", 0.0))
    fixed_delta_seconds = float(env.config.get("fixed_delta_seconds", 0.05))
    route_distance_m = max(progress_history) if progress_history else route_progress_from_mapping(metrics)
    metrics.update({
        "route_progress_m": route_progress_from_mapping(metrics),
        "ego_route_distance_m": float(route_distance_m),
        "progress_auc": progress_auc,
        "low_speed_ratio": low_speed_ratio,
        "low_risk_steps": int(low_risk_steps),
        "low_speed_steps": int(low_speed_steps),
        "max_speed": float(max(ego_speed_history)) if ego_speed_history else fallback_speed,
        "min_speed": float(min(ego_speed_history)) if ego_speed_history else fallback_speed,
        "combined_efficiency": 0.8 * progress_auc + 0.2 * (1.0 - low_speed_ratio),
        "fixed_delta_seconds": fixed_delta_seconds,
        "elapsed_time_s": float(step + 1) * fixed_delta_seconds,
        **behavior_marker_metrics(behavior_markers, step + 1),
    })
    return metrics


def run_ego_training_episode(env, mappo, ego, train_mod, args, seed):
    state, reset_info = env.reset(seed=seed)
    n_agents = len(env.controlled_vehicles)
    ego_state = np.asarray(env.obs2, dtype=np.float32).flatten()
    episode_reward = 0.0
    last_info = {"route_completion": reset_info.get("route_completion", 0.0)}
    diagnostics = train_mod.init_episode_diagnostics()
    progress_history = []
    ego_speed_history = []
    low_risk_steps = 0
    low_speed_steps = 0
    behavior_markers = init_behavior_markers()
    cruise_speed = float(env.config.get("target_speed_cruise", 20.0))
    for step in range(int(args.max_steps)):
        adv_actions = mappo.action(state, n_agents)
        ego_action, ego_log_prob, ego_value = ego.select_action(ego_state, deterministic=False)
        full_action = train_mod.append_hdv_actions(env, list(adv_actions) + [ego_action], args)
        next_state, global_reward, terminated, truncated, info = env.step(full_action)
        next_ego_state = np.asarray(env.obs2, dtype=np.float32).flatten()
        ego_reward = float(info.get("ego_reward", global_reward))
        done = bool(terminated or truncated)
        ego.step(ego_state, ego_action, ego_log_prob, ego_value, ego_reward, done, next_ego_state)
        episode_reward += ego_reward
        train_mod.update_episode_diagnostics(diagnostics, info)
        last_info = info
        progress_history.append(route_progress_from_mapping(info))
        ego_speed = float(getattr(getattr(env, "ego_vehicle", None), "speed", info.get("average_speed", 0.0)))
        ego_speed_history.append(ego_speed)
        risk = max(
            float(info.get("ego_obstacle_risk", 0.0)),
            float(info.get("ego_obstacle_risk_control", 0.0)),
            float(info.get("ego_target_lane_obstacle_risk", 0.0)),
        )
        safety_override = (
            bool(info.get("ego_obstacle_takeover_active", False))
            or bool(info.get("ego_obstacle_stop_active", False))
            or float(info.get("ego_safety_brake", 0.0)) > 1e-6
        )
        if risk < float(args.low_risk_threshold) and not safety_override:
            low_risk_steps += 1
            if ego_speed < float(args.low_speed_alpha) * cruise_speed:
                low_speed_steps += 1
        update_behavior_markers(behavior_markers, info, ego_action, ego_speed, cruise_speed, args)
        state = next_state
        ego_state = next_ego_state
        if done:
            break
    rewards = np.asarray(last_info.get("agents_rewards", [0.0]), dtype=np.float32)
    metrics = train_mod.episode_metrics(step + 1, episode_reward, rewards, train_mod.merge_episode_diagnostics(last_info, diagnostics))
    completed = bool(last_info.get("success", False)) or float(metrics.get("route_completion", 0.0)) >= 0.99
    progress_auc = calculate_progress_auc(
        progress_history,
        float(metrics.get("route_length", env.config.get("route_length", 200.0))),
        int(args.max_steps),
        completed,
    )
    low_speed_ratio = float(low_speed_steps) / max(float(low_risk_steps), 1.0)
    fallback_speed = float(metrics.get("average_speed", 0.0))
    fixed_delta_seconds = float(env.config.get("fixed_delta_seconds", 0.05))
    route_distance_m = max(progress_history) if progress_history else route_progress_from_mapping(metrics)
    metrics.update({
        "route_progress_m": route_progress_from_mapping(metrics),
        "ego_route_distance_m": float(route_distance_m),
        "progress_auc": progress_auc,
        "low_speed_ratio": low_speed_ratio,
        "low_risk_steps": int(low_risk_steps),
        "low_speed_steps": int(low_speed_steps),
        "max_speed": float(max(ego_speed_history)) if ego_speed_history else fallback_speed,
        "min_speed": float(min(ego_speed_history)) if ego_speed_history else fallback_speed,
        "combined_efficiency": 0.8 * progress_auc + 0.2 * (1.0 - low_speed_ratio),
        "fixed_delta_seconds": fixed_delta_seconds,
        "elapsed_time_s": float(step + 1) * fixed_delta_seconds,
        **behavior_marker_metrics(behavior_markers, step + 1),
    })
    return metrics


def evaluate_population(population, current_ego_checkpoint, frozen_egos, adv_lookup, args, seeds, output_dir, generation, runtime):
    train_mod, MAPPOAgent, EgoPPOAdapter, make_env = runtime
    env_rows = []
    policy_rows = []
    for spec in population:
        current_summary, current_rows = evaluate_policy_on_env(
            spec, current_ego_checkpoint, "current", "current", adv_lookup, args, seeds,
            train_mod, MAPPOAgent, EgoPPOAdapter, make_env,
        )
        policy_rows.extend({"generation": generation, **row} for row in current_rows)
        parent_count = 0
        near_count = 0
        best_mean_gap = 0.0
        best_better_seed_count = 0
        for frozen in frozen_egos:
            frozen_summary, frozen_rows = evaluate_policy_on_env(
                spec, frozen.checkpoint, "frozen", frozen.label, adv_lookup, args, seeds,
                train_mod, MAPPOAgent, EgoPPOAdapter, make_env,
            )
            policy_rows.extend({"generation": generation, **row} for row in frozen_rows)
            if current_summary["all_safe"] and frozen_summary["all_safe"]:
                ratio, better_seed_count = paired_efficiency_gap(current_rows, frozen_rows)
                best_mean_gap = max(best_mean_gap, ratio)
                best_better_seed_count = max(best_better_seed_count, better_seed_count)
                if ratio >= float(args.efficiency_margin) and better_seed_count >= int(args.better_seed_min_count):
                    parent_count += 1
                elif ratio > 0.0:
                    near_count += 1
        spec.parent_count = int(parent_count)
        spec.near_count = int(near_count)
        spec.score = float(parent_count) + 0.01 * float(near_count)
        env_rows.append({
            "generation": generation,
            **asdict(spec),
            "selected_as_parent": False,
            "current_all_safe": current_summary["all_safe"],
            "current_efficiency": current_summary["mean_efficiency"],
            "current_progress_auc": current_summary["mean_progress_auc"],
            "current_low_speed_ratio": current_summary["mean_low_speed_ratio"],
            "current_crash_rate": current_summary["crash_rate"],
            "current_route_completion": current_summary["route_completion"],
            "current_route_distance_m": current_summary["route_distance_m"],
            "current_avg_speed": current_summary["avg_speed"],
            "current_max_speed": current_summary["max_speed"],
            "current_min_speed": current_summary["min_speed"],
            "current_front_slow_ratio": current_summary["mean_front_slow_ratio"],
            "current_front_slow_episode_rate": current_summary["front_slow_episode_rate"],
            "current_safe_lane_change_opportunity_ratio": current_summary["mean_safe_lane_change_opportunity_ratio"],
            "current_safe_lane_change_opportunity_episode_rate": current_summary["safe_lane_change_opportunity_episode_rate"],
            "current_ego_lane_change_action_ratio": current_summary["mean_ego_lane_change_action_ratio"],
            "current_ego_lane_change_episode_rate": current_summary["ego_lane_change_episode_rate"],
            "current_ego_lane_change_success_episode_rate": current_summary["ego_lane_change_success_episode_rate"],
            "current_ego_overtake_success_episode_rate": current_summary["ego_overtake_success_episode_rate"],
            "best_frozen_relative_gap": best_mean_gap,
            "best_frozen_better_seed_count": best_better_seed_count,
        })
        print(f"  eval env={spec.environment_id:03d} parent_count={parent_count:02d} near={near_count:02d} current_eff={current_summary['mean_efficiency']:.3f}")
    write_csv(Path(output_dir) / "policy_eval_log.csv", policy_rows, mode="a")
    return env_rows


def select_parents(population):
    if not population:
        return []
    parent_count = max(1, int(np.ceil(len(population) / 5.0)))
    ranked = sorted(
        population,
        key=lambda item: (item.parent_count, item.near_count, item.score, -item.environment_id),
        reverse=True,
    )
    return ranked[:parent_count]


def update_population(population, parents, generation, templates, adv_steps, args, rng):
    candidates = [copy.deepcopy(item) for item in population]
    next_id = max(item.environment_id for item in candidates) + 1
    for parent in parents:
        for _ in range(int(args.children_per_parent)):
            candidates.append(mutate_environment(parent, next_id, generation, templates, adv_steps, rng))
            next_id += 1
    candidates = sorted(candidates, key=lambda item: (item.parent_count, item.near_count, -item.generation_created), reverse=True)
    selected = candidates[:int(args.population_size)]
    for idx, spec in enumerate(selected):
        spec.environment_id = idx
    return selected


def training_schedule(population, episodes, rng):
    q, r = divmod(int(episodes), len(population))
    schedule = []
    for _ in range(q):
        batch = list(population)
        rng.shuffle(batch)
        schedule.extend(batch)
    extras = sorted(population, key=lambda item: (item.parent_count, item.near_count), reverse=True)[:r]
    rng.shuffle(extras)
    schedule.extend(extras)
    return schedule


def effective_train_episodes_per_generation(args):
    if int(args.train_episodes_per_generation) > 0:
        return int(args.train_episodes_per_generation)
    return int(args.population_size) * int(args.train_episodes_per_env)


def serialize_population(population):
    return [asdict(item) for item in population]


def deserialize_population(rows):
    return [EnvironmentSpec(**row) for row in rows]


def encode_rng_state(rng):
    state = rng.get_state()
    return {
        "bit_generator": state[0],
        "state": state[1].tolist(),
        "pos": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def decode_rng_state(rng, data):
    if not data:
        return
    rng.set_state((
        data["bit_generator"],
        np.asarray(data["state"], dtype=np.uint32),
        int(data["pos"]),
        int(data["has_gauss"]),
        float(data["cached_gaussian"]),
    ))


def write_resume_state(output_dir, **state):
    state = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **state,
    }
    path = Path(output_dir) / "finetune_resume_state.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return path


def read_resume_state(output_dir):
    path = Path(output_dir) / "finetune_resume_state.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_row_value(rows, key, default=0.0):
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, default)))
        except (TypeError, ValueError):
            continue
    return float(np.mean(values)) if values else float(default)


def carla_restart_marker(total_ep, args):
    interval = int(getattr(args, "carla_restart_episode_interval", 0))
    if interval <= 0 or int(total_ep) <= 0:
        return 0
    return int(total_ep)


def carla_restart_due(total_ep, last_restart_marker, args):
    interval = int(getattr(args, "carla_restart_episode_interval", 0))
    if interval <= 0 or int(total_ep) <= 0:
        return False
    return int(total_ep) - int(last_restart_marker) >= interval


def write_carla_restart_state(output_dir, generation, train_index, episode, checkpoint_path, args, mode):
    state = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "generation": int(generation),
        "train_index": int(train_index),
        "ego_episode": int(episode),
        "checkpoint": str(checkpoint_path),
        "restart_interval": int(args.carla_restart_episode_interval),
        "mode": mode,
    }
    path = Path(output_dir) / "carla_restart_state.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def handle_carla_episode_restart(args, output_dir, generation, train_index, episode, checkpoint_path):
    command = str(getattr(args, "carla_restart_command", "") or "").strip()
    if command:
        write_carla_restart_state(output_dir, generation, train_index, episode, checkpoint_path, args, "command")
        print(f"[CARLA RESTART] ego_ep={episode} checkpoint={checkpoint_path}")
        print(f"[CARLA RESTART] running command: {command}")
        subprocess.run(command, shell=True, check=True)
        wait_seconds = float(getattr(args, "carla_restart_wait_seconds", 0.0))
        if wait_seconds > 0.0:
            print(f"[CARLA RESTART] waiting {wait_seconds:.1f}s before reconnect")
            time.sleep(wait_seconds)
        return

    write_carla_restart_state(output_dir, generation, train_index, episode, checkpoint_path, args, "exit")
    raise CarlaEpisodeRestartRequested(episode, checkpoint_path, int(args.carla_restart_exit_code))


def train_current_ego(
    population,
    current_ego_checkpoint,
    adv_lookup,
    args,
    output_dir,
    generation,
    runtime,
    rng,
    start_index=0,
    schedule_env_ids=None,
    resume_context=None,
):
    train_mod, MAPPOAgent, EgoPPOAdapter, make_env = runtime
    first_env = make_env(env_config_from_spec(population[0], args), config_path=args.config)
    try:
        ego = EgoPPOAdapter.from_config(train_mod.ego_config(args), first_env.n_s, first_env.n_a)
        ego.load(str(current_ego_checkpoint))
    finally:
        first_env.close()

    log_rows = []
    steps_since_update = 0
    losses = {}
    train_episodes = effective_train_episodes_per_generation(args)
    if schedule_env_ids:
        env_by_id = {int(item.environment_id): item for item in population}
        schedule = [env_by_id[int(env_id)] for env_id in schedule_env_ids]
    else:
        schedule = training_schedule(population, train_episodes, rng)
    start_index = max(0, min(int(start_index), len(schedule)))
    last_carla_restart_marker = carla_restart_marker(int(ego.policy.n_episodes), args)
    for idx, spec in enumerate(schedule[start_index:], start=start_index):
        adv_ckpt = adv_lookup[int(spec.adv_step)]
        env = make_env(env_config_from_spec(spec, args), config_path=args.config)
        try:
            mappo = MAPPOAgent.from_config({"reward_type": "agents_rewards", "use_cuda": not args.no_cuda, "torch_seed": args.torch_seed}, env.n_s, env.n_a)
            mappo.load(adv_ckpt.model_dir, adv_ckpt.step, train_mode=False)
            seed = int(args.seed) + generation * 100000 + idx
            metrics = run_ego_training_episode(env, mappo, ego, train_mod, args, seed)
            steps_since_update += int(metrics["steps"])
            if steps_since_update >= int(args.ego_n_steps):
                losses = ego.update()
                steps_since_update = 0
            ego.end_episode()
        finally:
            env.close()
        total_ep = int(ego.policy.n_episodes)
        log_rows.append({
            "generation": generation,
            "train_index": idx + 1,
            "ego_episode": total_ep,
            "environment_id": spec.environment_id,
            "adv_step": spec.adv_step,
            "template": spec.template,
            "combined_efficiency": metrics.get("combined_efficiency", ""),
            "progress_auc": metrics.get("progress_auc", ""),
            "ego_route_distance_m": metrics.get("ego_route_distance_m", ""),
            "route_progress_m": metrics.get("route_progress_m", ""),
            "low_speed_ratio": metrics.get("low_speed_ratio", ""),
            "low_risk_steps": metrics.get("low_risk_steps", ""),
            "low_speed_steps": metrics.get("low_speed_steps", ""),
            "front_slow_flag": metrics.get("front_slow_flag", ""),
            "front_slow_steps": metrics.get("front_slow_steps", ""),
            "front_slow_ratio": metrics.get("front_slow_ratio", ""),
            "safe_lane_change_opportunity_flag": metrics.get("safe_lane_change_opportunity_flag", ""),
            "safe_lane_change_opportunity_steps": metrics.get("safe_lane_change_opportunity_steps", ""),
            "safe_lane_change_opportunity_ratio": metrics.get("safe_lane_change_opportunity_ratio", ""),
            "ego_lane_change_action_flag": metrics.get("ego_lane_change_action_flag", ""),
            "ego_lane_change_action_steps": metrics.get("ego_lane_change_action_steps", ""),
            "ego_lane_change_action_ratio": metrics.get("ego_lane_change_action_ratio", ""),
            "ego_lane_change_success_flag": metrics.get("ego_lane_change_success_flag", ""),
            "ego_lane_change_success_steps": metrics.get("ego_lane_change_success_steps", ""),
            "ego_overtake_success_flag": metrics.get("ego_overtake_success_flag", ""),
            "ego_overtake_success_steps": metrics.get("ego_overtake_success_steps", ""),
            "ego_lane_change_intent_flag": metrics.get("ego_lane_change_intent_flag", ""),
            "ego_lane_change_intent_steps": metrics.get("ego_lane_change_intent_steps", ""),
            "avg_speed": metrics.get("average_speed", ""),
            "max_speed": metrics.get("max_speed", ""),
            "min_speed": metrics.get("min_speed", ""),
            **train_mod.episode_log_row(total_ep, metrics, losses),
        })
        if int(args.eval_interval) > 0 and (idx + 1) % int(args.eval_interval) == 0:
            print(f"  train gen={generation} ep={idx + 1:04d}/{len(schedule)} ego_ep={total_ep} reward={metrics['episode_reward']:.2f} crash={metrics['crash']}")
        if int(args.save_interval) > 0 and total_ep > 0 and total_ep % int(args.save_interval) == 0:
            ego.save(str(Path(output_dir) / "models" / "ego" / f"checkpoint-{total_ep}.pt"))
        if carla_restart_due(total_ep, last_carla_restart_marker, args):
            if steps_since_update > 0:
                losses = ego.update()
                steps_since_update = 0
            checkpoint_path = Path(output_dir) / "models" / "ego" / f"checkpoint-{total_ep}.pt"
            ego.save(str(checkpoint_path))
            write_csv(Path(output_dir) / "ego_train_log.csv", log_rows, mode="a")
            log_rows = []
            resume_context = resume_context or {}
            write_resume_state(
                output_dir,
                active=True,
                phase="training",
                generation=int(generation),
                train_index_next=int(idx + 1),
                current_ego_checkpoint=str(checkpoint_path),
                population=serialize_population(population),
                schedule_env_ids=[int(item.environment_id) for item in schedule],
                rng_state=encode_rng_state(rng),
                **resume_context,
            )
            handle_carla_episode_restart(args, output_dir, generation, idx + 1, total_ep, checkpoint_path)
            last_carla_restart_marker = carla_restart_marker(total_ep, args)
    if steps_since_update > 0:
        ego.update()
    final_ep = int(ego.policy.n_episodes)
    final_path = Path(output_dir) / "models" / "ego" / f"checkpoint-{final_ep}.pt"
    ego.save(str(final_path))
    write_csv(Path(output_dir) / "ego_train_log.csv", log_rows, mode="a")
    return final_path


def runtime_imports():
    from carla_evolution.agents.ego_ppo import EgoPPOAdapter
    from carla_evolution.agents.mappo import MAPPOAgent
    from carla_evolution.envs.factory import make_env
    from carla_evolution.training import train as train_mod
    return train_mod, MAPPOAgent, EgoPPOAdapter, make_env


def main():
    args = parse_args()
    templates = parse_str_list(args.scenario_presets)
    unknown = [item for item in templates if item not in SCENARIO_PRESETS]
    if unknown:
        raise ValueError(f"Unknown scenario preset(s): {unknown}")
    adv_steps = parse_int_list(args.adv_steps)
    selection_seeds = parse_int_list(args.selection_seeds)[:int(args.selection_episodes)]
    validation_seeds = parse_int_list(args.validation_seeds)[:int(args.validation_episodes)]
    search_dirs = [Path(item) for item in (args.ego_search_dir or [DEFAULT_RESULTS])]
    adv_search_dirs = [Path(item) for item in (args.adv_search_dir or [DEFAULT_RESULTS])]
    output_dir = make_output_dir(args)
    rng = np.random.RandomState(int(args.seed))
    resume_state = read_resume_state(output_dir) if args.resume_dir else None

    selected_ego_rows = []
    selected_adv_rows = []
    if int(args.frozen_ego_set_size) > 0:
        frozen_egos, adv_steps, selected_ego_rows, selected_adv_rows = select_experiment_assets(args, search_dirs, adv_search_dirs)
    else:
        frozen_egos = read_frozen_ego_set(args.frozen_ego_csv, search_dirs)
    current_ego = Path(args.current_ego_checkpoint) if args.current_ego_checkpoint else latest_ego_checkpoint(search_dirs)
    if resume_state:
        current_ego = Path(resume_state.get("current_ego_checkpoint") or current_ego)
        if "population" in resume_state:
            population = deserialize_population(resume_state["population"])
        else:
            population_path = Path(output_dir) / f"population_gen{int(resume_state.get('generation', 0))}.json"
            with population_path.open("r", encoding="utf-8") as f:
                population = deserialize_population(json.load(f))
        decode_rng_state(rng, resume_state.get("rng_state"))
    else:
        population = initialize_population(args.population_size, templates, adv_steps, args, rng)
    adv_checkpoints = resolve_adv_checkpoints(adv_steps, adv_search_dirs)
    adv_lookup = {item.step: item for item in adv_checkpoints}
    hdv_path = latest_hdv_model(DEFAULT_RESULTS) if args.hdv_model == "latest" else (Path(args.hdv_model) if str(args.hdv_model).strip() else None)
    train_episodes_per_generation = effective_train_episodes_per_generation(args)

    if not resume_state:
        manifest = {
            "current_ego_checkpoint": str(current_ego),
            "frozen_ego_count": len(frozen_egos),
            "frozen_ego_set_size": int(args.frozen_ego_set_size),
            "experiment_candidates_csv": str(args.experiment_candidates_csv) if int(args.frozen_ego_set_size) > 0 else "",
            "prefer_ego_round_after": int(args.prefer_ego_round_after),
            "adv_steps": adv_steps,
            "adv_checkpoints": [asdict(item) for item in adv_checkpoints],
            "matched_adv_init": int(args.frozen_ego_set_size) > 0 and not args.disable_matched_adv_init,
            "effective_train_episodes_per_generation": train_episodes_per_generation,
            "train_episodes_per_env": int(args.train_episodes_per_env),
            "hdv_model": str(hdv_path or ""),
            "selection_seeds": selection_seeds,
            "validation_seeds": validation_seeds,
            "args": vars(args),
        }
        with (Path(output_dir) / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        with (Path(output_dir) / "population_gen0.json").open("w", encoding="utf-8") as f:
            json.dump([asdict(item) for item in population], f, ensure_ascii=False, indent=2)
        if selected_ego_rows:
            write_csv(Path(output_dir) / "selected_frozen_egos.csv", selected_ego_rows)
        if selected_adv_rows:
            write_csv(Path(output_dir) / "selected_initial_adv_models.csv", selected_adv_rows)

    print("=" * 80)
    print("Enhanced-POET ego fine-tuning")
    print(f"  output_dir:       {output_dir}")
    print(f"  current_ego:      {current_ego}")
    print(f"  frozen_ego_count: {len(frozen_egos)}")
    print(f"  adv_steps:        {adv_steps}")
    if selected_adv_rows:
        print(f"  matched_adv_init: {len(selected_adv_rows)} initial rows")
    print(f"  hdv_model:        {hdv_path or 'disabled'}")
    print(f"  population_size:  {len(population)}")
    print(f"  train_episodes:   {train_episodes_per_generation} ({args.train_episodes_per_env}/env)")
    print(f"  generations:      {args.generations}")
    if int(args.carla_restart_episode_interval) > 0:
        mode = "command" if str(args.carla_restart_command).strip() else f"exit_code={args.carla_restart_exit_code}"
        print(f"  carla_restart:    every {args.carla_restart_episode_interval} ego episodes ({mode})")
    print("=" * 80)
    if args.dry_run:
        for spec in population[: min(10, len(population))]:
            print(f"[DRY-RUN] env={spec.environment_id:03d} template={spec.template} adv={spec.adv_step} num_adv={spec.num_adv} num_hdv={spec.num_natural} slots={spec.adv_slots}")
        print(f"[DRY-RUN] manifests written under {output_dir}")
        return

    runtime = runtime_imports()
    train_mod = runtime[0]
    args._hdv_policy, resolved_hdv = load_hdv(args, train_mod)
    if resolved_hdv:
        print(f"[HDV] loaded {resolved_hdv}")

    no_large_diff_generations = int((resume_state or {}).get("no_large_diff_generations", 0))
    no_improvement_generations = int((resume_state or {}).get("no_improvement_generations", 0))
    best_validation_efficiency = (resume_state or {}).get("best_validation_efficiency", None)
    start_generation = int((resume_state or {}).get("generation", 0))
    resume_phase = str((resume_state or {}).get("phase", ""))
    resume_train_index = int((resume_state or {}).get("train_index_next", 0))
    resume_schedule_env_ids = (resume_state or {}).get("schedule_env_ids")

    for generation in range(start_generation, int(args.generations)):
        if resume_state and generation == start_generation and resume_phase == "training":
            parents = select_parents(population)
            parent_ids = {item.environment_id for item in parents}
            max_parent_count = int((resume_state or {}).get("max_parent_count", max((item.parent_count for item in population), default=0)))
            selection_current_efficiency = float((resume_state or {}).get("selection_current_efficiency", 0.0))
            print(f"\n[GEN {generation}] resuming training at train_index={resume_train_index}")
            start_index = resume_train_index
            schedule_env_ids = resume_schedule_env_ids
        else:
            print(f"\n[GEN {generation}] evaluating population")
            env_rows = evaluate_population(population, current_ego, frozen_egos, adv_lookup, args, selection_seeds, output_dir, generation, runtime)
            parents = select_parents(population)
            parent_ids = {item.environment_id for item in parents}
            for row in env_rows:
                row["selected_as_parent"] = row["environment_id"] in parent_ids
            write_csv(Path(output_dir) / "environment_log.csv", env_rows, mode="a")
            max_parent_count = max(item.parent_count for item in population)
            selection_current_efficiency = mean_row_value(env_rows, "current_efficiency")
            if max_parent_count <= 0:
                no_large_diff_generations += 1
            else:
                no_large_diff_generations = 0
            print(f"[GEN {generation}] parents={sorted(parent_ids)} max_count={max_parent_count}")
            start_index = 0
            schedule_env_ids = None

        print(f"[GEN {generation}] training current ego")
        current_ego = train_current_ego(
            population,
            current_ego,
            adv_lookup,
            args,
            output_dir,
            generation,
            runtime,
            rng,
            start_index=start_index,
            schedule_env_ids=schedule_env_ids,
            resume_context={
                "no_large_diff_generations": int(no_large_diff_generations),
                "no_improvement_generations": int(no_improvement_generations),
                "best_validation_efficiency": best_validation_efficiency,
                "parent_ids": sorted(int(item) for item in parent_ids),
                "max_parent_count": int(max_parent_count),
                "selection_current_efficiency": float(selection_current_efficiency),
            },
        )
        resume_state = None
        resume_phase = ""
        resume_train_index = 0
        resume_schedule_env_ids = None

        print(f"[GEN {generation}] validation on parent environments")
        validation_population = parents[: max(1, min(len(parents), 5))]
        validation_rows = evaluate_population(validation_population, current_ego, [], adv_lookup, args, validation_seeds, output_dir, generation, runtime)
        write_csv(Path(output_dir) / "validation_environment_log.csv", validation_rows, mode="a")
        validation_efficiency = mean_row_value(validation_rows, "current_efficiency")
        validation_route_distance_m = mean_row_value(validation_rows, "current_route_distance_m")
        validation_avg_speed = mean_row_value(validation_rows, "current_avg_speed")
        validation_max_speed = max((float(row.get("current_max_speed", 0.0)) for row in validation_rows), default=0.0)
        validation_min_speed = min((float(row.get("current_min_speed", 0.0)) for row in validation_rows), default=0.0)
        validation_front_slow_ratio = mean_row_value(validation_rows, "current_front_slow_ratio")
        validation_front_slow_episode_rate = mean_row_value(validation_rows, "current_front_slow_episode_rate")
        validation_safe_lane_change_opportunity_ratio = mean_row_value(validation_rows, "current_safe_lane_change_opportunity_ratio")
        validation_safe_lane_change_opportunity_episode_rate = mean_row_value(validation_rows, "current_safe_lane_change_opportunity_episode_rate")
        validation_ego_lane_change_action_ratio = mean_row_value(validation_rows, "current_ego_lane_change_action_ratio")
        validation_ego_lane_change_episode_rate = mean_row_value(validation_rows, "current_ego_lane_change_episode_rate")
        validation_ego_lane_change_success_episode_rate = mean_row_value(validation_rows, "current_ego_lane_change_success_episode_rate")
        validation_ego_overtake_success_episode_rate = mean_row_value(validation_rows, "current_ego_overtake_success_episode_rate")
        if best_validation_efficiency is None or validation_efficiency > best_validation_efficiency + float(args.improvement_min_delta):
            best_validation_efficiency = validation_efficiency
            no_improvement_generations = 0
        else:
            no_improvement_generations += 1

        stop_reason = ""
        if no_large_diff_generations >= int(args.no_large_diff_patience):
            stop_reason = (
                f"no_large_efficiency_difference_for_{no_large_diff_generations}_generations"
            )
        elif int(args.no_improvement_patience) > 0 and no_improvement_generations >= int(args.no_improvement_patience):
            stop_reason = (
                f"validation_efficiency_no_improvement_for_{no_improvement_generations}_generations"
            )

        write_csv(Path(output_dir) / "generation_summary.csv", [{
            "generation": generation,
            "max_parent_count": max_parent_count,
            "selected_parent_count": len(parents),
            "parent_ids": json.dumps(sorted(parent_ids), ensure_ascii=False),
            "selection_current_efficiency": selection_current_efficiency,
            "validation_current_efficiency": validation_efficiency,
            "validation_ego_route_distance_m": validation_route_distance_m,
            "validation_ego_avg_speed": validation_avg_speed,
            "validation_ego_max_speed": validation_max_speed,
            "validation_ego_min_speed": validation_min_speed,
            "validation_front_slow_ratio": validation_front_slow_ratio,
            "validation_front_slow_episode_rate": validation_front_slow_episode_rate,
            "validation_safe_lane_change_opportunity_ratio": validation_safe_lane_change_opportunity_ratio,
            "validation_safe_lane_change_opportunity_episode_rate": validation_safe_lane_change_opportunity_episode_rate,
            "validation_ego_lane_change_action_ratio": validation_ego_lane_change_action_ratio,
            "validation_ego_lane_change_episode_rate": validation_ego_lane_change_episode_rate,
            "validation_ego_lane_change_success_episode_rate": validation_ego_lane_change_success_episode_rate,
            "validation_ego_overtake_success_episode_rate": validation_ego_overtake_success_episode_rate,
            "best_validation_efficiency": best_validation_efficiency,
            "no_large_diff_generations": no_large_diff_generations,
            "no_improvement_generations": no_improvement_generations,
            "early_stop": bool(stop_reason),
            "stop_reason": stop_reason,
            "current_ego_checkpoint": str(current_ego),
        }], mode="a")

        if stop_reason:
            print(f"[EARLY STOP] {stop_reason}")
            break

        population = update_population(population, parents, generation + 1, templates, adv_steps, args, rng)
        with (Path(output_dir) / f"population_gen{generation + 1}.json").open("w", encoding="utf-8") as f:
            json.dump([asdict(item) for item in population], f, ensure_ascii=False, indent=2)
        write_resume_state(
            output_dir,
            active=False,
            phase="generation_complete",
            generation=int(generation + 1),
            train_index_next=0,
            current_ego_checkpoint=str(current_ego),
            population=serialize_population(population),
            rng_state=encode_rng_state(rng),
            no_large_diff_generations=int(no_large_diff_generations),
            no_improvement_generations=int(no_improvement_generations),
            best_validation_efficiency=best_validation_efficiency,
        )

    print(f"\nComplete. Latest fine-tuned ego checkpoint: {current_ego}")


if __name__ == "__main__":
    try:
        main()
    except CarlaEpisodeRestartRequested as exc:
        print(
            "\n[CARLA RESTART REQUESTED] "
            f"Saved checkpoint-{exc.episode}.pt at {exc.checkpoint_path}. "
            "Restart CARLA, then rerun the same training command to resume from the latest checkpoint."
        )
        sys.exit(exc.exit_code)
