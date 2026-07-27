#!/usr/bin/env python3
"""Evaluate one frozen EgoPPO checkpoint against sampled frozen MAPPO checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = MODEL_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from carla_evolution.scripts import finetune_ego_enhanced_poet as finetune


DEFAULT_RESULTS = MODEL_ROOT / "results"
DEFAULT_CONFIG = MODEL_ROOT / "configs" / "carla_0915.yaml"
DEFAULT_ADV_MODEL_DIR = DEFAULT_RESULTS / "joint_Jul_01_14_59_59" / "models" / "adv"
DEFAULT_JOINT_ROUND_LOG = DEFAULT_RESULTS / "joint_Jul_01_14_59_59" / "joint_round_log.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "carla"], default="carla")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-dir", default=str(DEFAULT_RESULTS / "ego_adv_frozen_eval"))
    parser.add_argument(
        "--ego-checkpoint",
        default=None,
        help="Frozen EgoPPO checkpoint to evaluate. Defaults to latest checkpoint-N.pt under --ego-search-dir/results.",
    )
    parser.add_argument("--ego-search-dir", action="append", default=None)
    parser.add_argument("--adv-model-dir", default=str(DEFAULT_ADV_MODEL_DIR))
    parser.add_argument(
        "--joint-round-log",
        default=str(DEFAULT_JOINT_ROUND_LOG),
        help="Joint round log used to map training round to adversarial checkpoint step.",
    )
    parser.add_argument("--adv-search-dir", action="append", default=None)
    parser.add_argument("--adv-round-last", type=int, default=60, help="Use the latest N training rounds as the sampling pool.")
    parser.add_argument(
        "--adv-pool-last",
        type=int,
        default=60,
        help="Fallback only: use the latest N paired actor/critic checkpoints if --joint-round-log is unavailable.",
    )
    parser.add_argument("--adv-sample-count", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=669)
    parser.add_argument("--eval-seeds", default="101,102,103", help="Used when --eval-episodes-per-adv <= 0.")
    parser.add_argument(
        "--eval-episodes-per-adv",
        type=int,
        default=0,
        help="If >0, generate this many distinct seeds for each adversarial checkpoint.",
    )
    parser.add_argument("--eval-seed-base", type=int, default=101)
    parser.add_argument("--eval-seed-stride", type=int, default=100000)
    parser.add_argument("--scenario-preset", default="slow_front_left_escape")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-adv", type=int, default=3)
    parser.add_argument("--num-natural", type=int, default=0)
    parser.add_argument("--spawn-start-index", type=int, default=0)
    parser.add_argument("--randomize-scenarios", action="store_true", default=False)
    parser.add_argument("--relative-offset-jitter", type=float, default=0.0)
    parser.add_argument("--randomize-lane-offsets", action="store_true", default=False)
    parser.add_argument("--randomize-lane-offset-min", type=int, default=0)
    parser.add_argument("--randomize-lane-offset-max", type=int, default=2)
    parser.add_argument("--randomize-spawn-start-index", action="store_true", default=False)
    parser.add_argument("--shuffle-spawn-points", action="store_true", default=False)
    parser.add_argument("--target-speed-cruise", type=float, default=None)
    parser.add_argument("--route-length", type=float, default=None)
    parser.add_argument("--hdv-model", default="", help="'latest', explicit SB3 .zip path, or empty string to disable natural-car policy model.")
    parser.add_argument("--hdv-action", default="keep_lane")
    parser.add_argument("--low-speed-alpha", type=float, default=0.5)
    parser.add_argument("--low-risk-threshold", type=float, default=0.25)
    parser.add_argument("--front-slow-alpha", type=float, default=0.6)
    parser.add_argument("--front-slow-gap", type=float, default=35.0)
    parser.add_argument("--torch-seed", type=int, default=669)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--carla-rpc-timeout", type=float, default=180.0)
    parser.add_argument("--purge-existing-actors-on-reset", action="store_true", default=False)
    parser.add_argument("--cleanup-destroy-mode", choices=["sequential", "batch"], default="sequential")
    parser.add_argument("--render", action="store_true", default=False)
    return parser.parse_args()


def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def seeds_for_adv(args, adv_index: int):
    if int(args.eval_episodes_per_adv) > 0:
        base = int(args.eval_seed_base) + int(adv_index) * int(args.eval_seed_stride)
        return [base + idx for idx in range(int(args.eval_episodes_per_adv))]
    seeds = parse_int_list(args.eval_seeds)
    if not seeds:
        raise ValueError("--eval-seeds cannot be empty when --eval-episodes-per-adv <= 0")
    return seeds


def paired_adv_steps(model_dir: Path):
    regex = re.compile(r"actor_(\d+)\.pt$")
    steps = []
    if not model_dir.exists():
        raise FileNotFoundError(f"Adv model dir not found: {model_dir}")
    for actor in model_dir.glob("actor_*.pt"):
        match = regex.fullmatch(actor.name)
        if not match:
            continue
        step = int(match.group(1))
        if (model_dir / f"critic_{step}.pt").exists():
            steps.append(step)
    steps = sorted(set(steps))
    if not steps:
        raise FileNotFoundError(f"No paired actor/critic checkpoints found under {model_dir}")
    return steps


def round_adv_boundaries(round_log: Path):
    if not round_log.exists():
        return []
    rows = []
    with round_log.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                round_no = int(float(row.get("round", 0)))
                adv_step = int(float(row.get("total_adversarial_episodes", 0)))
            except (TypeError, ValueError):
                continue
            if adv_step <= 0:
                continue
            rows.append({"round": round_no, "adv_step": adv_step})
    rows.sort(key=lambda item: item["round"])
    return rows


def round_for_adv_step(boundaries, adv_step):
    for row in boundaries:
        if int(adv_step) <= int(row["adv_step"]):
            return int(row["round"])
    return int(boundaries[-1]["round"]) if boundaries else ""


def checkpoint_rows_from_last_rounds(round_log: Path, model_dir: Path, last_rounds: int):
    boundaries = round_adv_boundaries(round_log)
    if not boundaries:
        return [], {}
    selected_boundaries = boundaries[-int(last_rounds):]
    start_round = int(selected_boundaries[0]["round"])
    end_round = int(selected_boundaries[-1]["round"])
    lower_step = 0
    for row in boundaries:
        if int(row["round"]) < start_round:
            lower_step = int(row["adv_step"])
        else:
            break
    upper_step = int(selected_boundaries[-1]["adv_step"])
    steps = paired_adv_steps(model_dir)
    rows = [
        {"round": round_for_adv_step(boundaries, step), "adv_step": int(step)}
        for step in steps
        if lower_step < int(step) <= upper_step
    ]
    meta = {
        "start_round": start_round,
        "end_round": end_round,
        "lower_step_exclusive": lower_step,
        "upper_step_inclusive": upper_step,
        "available_checkpoint_count": len(rows),
    }
    return rows, meta


def stratified_sample(values, sample_count, seed):
    values = sorted(values)
    sample_count = int(sample_count)
    if sample_count <= 0:
        return []
    if sample_count >= len(values):
        return values
    rng = np.random.RandomState(int(seed))
    bins = np.array_split(np.asarray(values, dtype=np.int64), sample_count)
    sampled = []
    for bin_values in bins:
        if len(bin_values) == 0:
            continue
        sampled.append(int(bin_values[int(rng.randint(0, len(bin_values)))]))
    return sorted(set(sampled))


def stratified_sample_rows(rows, sample_count, seed):
    rows = sorted(rows, key=lambda item: item["round"])
    sample_count = int(sample_count)
    if sample_count <= 0:
        return []
    if sample_count >= len(rows):
        return rows
    rng = np.random.RandomState(int(seed))
    bins = np.array_split(np.asarray(rows, dtype=object), sample_count)
    sampled = []
    for bin_values in bins:
        if len(bin_values) == 0:
            continue
        sampled.append(dict(bin_values[int(rng.randint(0, len(bin_values)))]))
    sampled.sort(key=lambda item: item["round"])
    return sampled


def make_output_dir(args):
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
        output_dir = Path(args.base_dir) / f"ego_adv_eval_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
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


def env_args_for_finetune(args):
    return SimpleNamespace(
        backend=args.backend,
        config=args.config,
        max_steps=int(args.max_steps),
        spawn_start_index=int(args.spawn_start_index),
        carla_rpc_timeout=float(args.carla_rpc_timeout),
        render=bool(args.render),
        hdv_action=args.hdv_action,
        _hdv_policy=None,
        randomize_scenarios=bool(args.randomize_scenarios),
        relative_offset_jitter=float(args.relative_offset_jitter),
        randomize_lane_offsets=bool(args.randomize_lane_offsets),
        randomize_lane_offset_min=int(args.randomize_lane_offset_min),
        randomize_lane_offset_max=int(args.randomize_lane_offset_max),
        randomize_spawn_start_index=bool(args.randomize_spawn_start_index),
        shuffle_spawn_points=bool(args.shuffle_spawn_points),
        low_speed_alpha=float(args.low_speed_alpha),
        low_risk_threshold=float(args.low_risk_threshold),
        front_slow_alpha=float(args.front_slow_alpha),
        front_slow_gap=float(args.front_slow_gap),
        no_cuda=bool(args.no_cuda),
        torch_seed=int(args.torch_seed),
        min_num_adv=int(args.num_adv),
        max_num_adv=int(args.num_adv),
        purge_existing_actors_on_reset=bool(args.purge_existing_actors_on_reset),
        cleanup_destroy_mode=str(args.cleanup_destroy_mode),
    )


def make_spec(adv_step, args):
    if args.scenario_preset not in finetune.SCENARIO_PRESETS:
        raise ValueError(f"Unknown scenario preset: {args.scenario_preset}")
    rng = np.random.RandomState(0)
    spec = finetune.make_env_spec(
        env_id=int(adv_step),
        template=args.scenario_preset,
        adv_step=int(adv_step),
        num_adv=int(args.num_adv),
        rng=rng,
    )
    spec.num_natural = int(args.num_natural)
    if args.target_speed_cruise is not None:
        spec.target_speed_cruise = float(args.target_speed_cruise)
    if args.route_length is not None:
        spec.route_length = float(args.route_length)
    return spec


def episode_row(ego_checkpoint, adv_step, adv_model_dir, seed, metrics, train_mod):
    losses = {}
    row = {
        "ego_checkpoint": str(ego_checkpoint),
        "adv_step": int(adv_step),
        "adv_model_dir": str(adv_model_dir),
        "seed": int(seed),
        "combined_efficiency": float(metrics.get("combined_efficiency", 0.0)),
        "traffic_efficiency": finetune.efficiency_from_metrics(metrics),
        "progress_auc": float(metrics.get("progress_auc", 0.0)),
        "route_speed_efficiency": finetune.route_speed_efficiency_from_metrics(metrics),
        "route_completion": metrics.get("route_completion", 0.0),
        "route_progress_m": metrics.get("route_progress_m", finetune.route_progress_from_mapping(metrics)),
        "ego_route_distance_m": metrics.get("ego_route_distance_m", metrics.get("route_progress_m", 0.0)),
        "completion_horizon_m": metrics.get("route_completion_distance", 0.0),
        "elapsed_steps": metrics.get("steps", 0),
        "elapsed_time_s": float(metrics.get("elapsed_time_s", float(metrics.get("steps", 0)) * finetune.metrics_dt(metrics))),
        "fixed_delta_seconds": finetune.metrics_dt(metrics),
        "mean_speed": metrics.get("average_speed", 0.0),
        "max_speed": metrics.get("max_speed", metrics.get("average_speed", 0.0)),
        "min_speed": metrics.get("min_speed", metrics.get("average_speed", 0.0)),
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
    row.update(train_mod.episode_log_row(0, metrics, losses))
    return row


def summarize_adv_rows(rows):
    numeric_keys = [
        "combined_efficiency",
        "traffic_efficiency",
        "progress_auc",
        "route_speed_efficiency",
        "route_completion",
        "route_progress_m",
        "ego_route_distance_m",
        "completion_horizon_m",
        "elapsed_steps",
        "elapsed_time_s",
        "mean_speed",
        "max_speed",
        "min_speed",
        "low_speed_ratio",
        "low_risk_steps",
        "low_speed_steps",
        "front_slow_steps",
        "front_slow_ratio",
        "safe_lane_change_opportunity_steps",
        "safe_lane_change_opportunity_ratio",
        "ego_lane_change_action_steps",
        "ego_lane_change_action_ratio",
        "ego_lane_change_success_steps",
        "ego_overtake_success_steps",
        "episode_reward",
        "mean_agent_reward",
    ]
    bool_keys = [
        "crash",
        "agents_crash",
        "front_slow_flag",
        "safe_lane_change_opportunity_flag",
        "ego_lane_change_action_flag",
        "ego_lane_change_success_flag",
        "ego_overtake_success_flag",
    ]
    summary = {
        "adv_step": int(rows[0]["adv_step"]),
        "adv_model_dir": rows[0]["adv_model_dir"],
        "episodes": len(rows),
    }
    for key in numeric_keys:
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
    for key in bool_keys:
        values = [float(bool(row.get(key, False))) for row in rows]
        summary[f"{key}_rate"] = float(np.mean(values)) if values else 0.0
    return summary


def main():
    args = parse_args()
    env_args = env_args_for_finetune(args)
    train_mod, MAPPOAgent, EgoPPOAdapter, make_env = finetune.runtime_imports()

    ego_search_dirs = [Path(item) for item in (args.ego_search_dir or [DEFAULT_RESULTS])]
    ego_checkpoint = Path(args.ego_checkpoint) if args.ego_checkpoint else finetune.latest_ego_checkpoint(ego_search_dirs)
    if not ego_checkpoint.is_file():
        raise FileNotFoundError(f"Ego checkpoint not found: {ego_checkpoint}")

    adv_model_dir = Path(args.adv_model_dir)
    adv_pool_rows, round_pool_meta = checkpoint_rows_from_last_rounds(
        Path(args.joint_round_log),
        adv_model_dir,
        int(args.adv_round_last),
    )
    if adv_pool_rows:
        sampled_adv_rows = stratified_sample_rows(adv_pool_rows, int(args.adv_sample_count), int(args.sample_seed))
        sampling_mode = "round_log"
    else:
        adv_steps = paired_adv_steps(adv_model_dir)
        adv_pool = adv_steps[-int(args.adv_pool_last):]
        sampled_adv_rows = [
            {"round": "", "adv_step": step}
            for step in stratified_sample(adv_pool, int(args.adv_sample_count), int(args.sample_seed))
        ]
        sampling_mode = "checkpoint_file_fallback"
        round_pool_meta = {}
    sampled_adv_steps = [int(row["adv_step"]) for row in sampled_adv_rows]
    seed_plan = {
        int(row["adv_step"]): seeds_for_adv(args, idx)
        for idx, row in enumerate(sampled_adv_rows)
    }

    if str(args.hdv_model).strip():
        hdv_path = finetune.latest_hdv_model(DEFAULT_RESULTS) if args.hdv_model == "latest" else Path(args.hdv_model)
        env_args._hdv_policy = train_mod.load_hdv_policy(str(hdv_path))
    else:
        hdv_path = None

    output_dir = make_output_dir(args)
    manifest = {
        "ego_checkpoint": str(ego_checkpoint),
        "adv_model_dir": str(adv_model_dir),
        "joint_round_log": str(args.joint_round_log),
        "sampling_mode": sampling_mode,
        "adv_round_last": int(args.adv_round_last),
        "adv_pool_last": int(args.adv_pool_last),
        "round_pool": round_pool_meta,
        "sampled_min_round": int(min([row["round"] for row in sampled_adv_rows if row["round"] != ""], default=0)),
        "sampled_max_round": int(max([row["round"] for row in sampled_adv_rows if row["round"] != ""], default=0)),
        "adv_pool_min_step": int(min(sampled_adv_steps)),
        "adv_pool_max_step": int(max(sampled_adv_steps)),
        "sampled_adv_rows": sampled_adv_rows,
        "sampled_adv_steps": sampled_adv_steps,
        "eval_seeds": parse_int_list(args.eval_seeds),
        "eval_episodes_per_adv": int(args.eval_episodes_per_adv),
        "eval_seed_base": int(args.eval_seed_base),
        "eval_seed_stride": int(args.eval_seed_stride),
        "seed_plan": seed_plan,
        "scenario_preset": args.scenario_preset,
        "num_adv": int(args.num_adv),
        "num_natural": int(args.num_natural),
        "hdv_model": str(hdv_path or ""),
        "args": vars(args),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    all_rows = []
    summary_rows = []
    sampled_rows = []
    for adv_index, sampled_adv_row in enumerate(sampled_adv_rows):
        adv_step = int(sampled_adv_row["adv_step"])
        seeds = seed_plan[int(adv_step)]
        adv_ckpt = finetune.AdvCheckpoint(step=int(adv_step), model_dir=str(adv_model_dir))
        spec = make_spec(adv_step, args)
        sampled_rows.append({"round": sampled_adv_row.get("round", ""), "adv_step": int(adv_step), **asdict(spec)})
        env = make_env(finetune.env_config_from_spec(spec, env_args), config_path=args.config)
        rows_for_adv = []
        try:
            if int(args.num_adv) > 0:
                mappo = MAPPOAgent.from_config(
                    {"reward_type": "agents_rewards", "use_cuda": not args.no_cuda, "torch_seed": args.torch_seed},
                    env.n_s,
                    env.n_a,
                )
                mappo.load(adv_ckpt.model_dir, adv_ckpt.step, train_mode=False)
            else:
                mappo = None
            ego = EgoPPOAdapter.from_config({"use_cuda": not args.no_cuda, "seed": args.torch_seed}, env.n_s, env.n_a)
            ego.load(str(ego_checkpoint))
            for seed in seeds:
                metrics = finetune.run_eval_episode(env, mappo, ego, train_mod, env_args, int(seed))
                row = episode_row(ego_checkpoint, adv_step, adv_model_dir, int(seed), metrics, train_mod)
                row["round"] = sampled_adv_row.get("round", "")
                rows_for_adv.append(row)
                all_rows.append(row)
                print(
                    f"round={sampled_adv_row.get('round', '')} adv={adv_step} seed={seed} "
                    f"eff={row['combined_efficiency']:.3f} "
                    f"speed={float(row['mean_speed']):.2f} "
                    f"dist={float(row['ego_route_distance_m']):.1f} "
                    f"crash={row.get('crash', False)}"
                )
        finally:
            env.close()
        summary = summarize_adv_rows(rows_for_adv)
        summary["round"] = sampled_adv_row.get("round", "")
        summary_rows.append(summary)
        write_csv(output_dir / "episode_log.csv", all_rows)
        write_csv(output_dir / "adv_summary.csv", summary_rows)

    write_csv(output_dir / "sampled_adv_checkpoints.csv", sampled_rows)
    completion = {
        "status": "complete",
        "episode_count": int(len(all_rows)),
        "adv_count": int(len(summary_rows)),
        "sampled_adv_steps": sampled_adv_steps,
    }
    with (output_dir / "eval_complete.json").open("w", encoding="utf-8") as f:
        json.dump(completion, f, ensure_ascii=False, indent=2)
    print("=" * 80)
    print(f"Output dir: {output_dir}")
    print(f"Episode log: {output_dir / 'episode_log.csv'}")
    print(f"Summary log: {output_dir / 'adv_summary.csv'}")
    print(f"Sampled adv: {sampled_adv_steps}")


if __name__ == "__main__":
    main()
