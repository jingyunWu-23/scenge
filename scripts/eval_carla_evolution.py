"""Run CARLA Evolution natural-traffic and frozen-MAPPO evaluations.

This is a thin launcher around the existing ``carla_evolution`` evaluation
scripts. It keeps the ScenGE repository from vendoring the MAPPO/environment
implementation while still giving one stable entry point for the two migrated
experiments described in ``自然车与对抗车测试场景迁移说明.md``.

Examples:

    python scripts/eval_carla_evolution.py normal \\
        --ego-checkpoint /path/to/ego/checkpoint.pt

    python scripts/eval_carla_evolution.py adv \\
        --ego-checkpoint /path/to/ego/checkpoint.pt

    python scripts/eval_carla_evolution.py adv-sweep \\
        --ego-dir /path/to/ego/checkpoint_dir --include-final
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARLA_EVOLUTION_ROOT = Path(
    "/home/chenyuanwan/download/co-training/code-migration/模型/carla_evolution"
)
DEFAULT_ADV_RUN = "joint_Jul_01_14_59_59"
DEFAULT_ADV_ROUND_LAST = 60
DEFAULT_ADV_SAMPLE_COUNT = 10
DEFAULT_EVAL_SEED_BASE = 101
DEFAULT_EVAL_SEED_STRIDE = 100000
DEFAULT_MAX_STEPS = 200
DEFAULT_NUM_NATURAL = 3
DEFAULT_NUM_ADV = 3
DEFAULT_RELATIVE_OFFSET_JITTER = 10.0
DEFAULT_LANE_OFFSET_MIN = 0
DEFAULT_LANE_OFFSET_MAX = 2


def _path(value: str | Path) -> str:
    return str(Path(value).expanduser())


def _append_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def _append_repeat(cmd: list[str], flag: str, values: Sequence[str] | None) -> None:
    for value in values or ():
        cmd.extend([flag, value])


def _target_root_and_config(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.carla_evolution_root).expanduser().resolve()
    config = Path(args.config).expanduser()
    if not config.is_absolute():
        config = root / config
    return root, config


def _shared_target_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    root, config = _target_root_and_config(args)
    adv_model_dir = Path(args.adv_model_dir).expanduser()
    if not adv_model_dir.is_absolute():
        adv_model_dir = root / adv_model_dir
    joint_round_log = Path(args.joint_round_log).expanduser()
    if not joint_round_log.is_absolute():
        joint_round_log = root / joint_round_log
    return root, config, adv_model_dir, joint_round_log


def _validate_target(root: Path, script_name: str) -> Path:
    script_path = root / "scripts" / script_name
    missing = [
        path
        for path in (root, script_path, root / "agents" / "mappo.py", root / "envs" / "env.py")
        if not path.exists()
    ]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise SystemExit(f"Missing required carla_evolution files:\n  {formatted}")
    return script_path


def _run(cmd: Sequence[str], cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print(">>", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _base_env(args: argparse.Namespace, root: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(root), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if getattr(args, "cuda_visible_devices", None):
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def _append_common_scene_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--scenario-preset",
            args.scenario_preset,
            "--max-steps",
            str(args.max_steps),
            "--eval-seed-base",
            str(args.eval_seed_base),
            "--num-natural",
            str(args.num_natural),
            "--spawn-start-index",
            str(args.spawn_start_index),
            "--relative-offset-jitter",
            str(args.relative_offset_jitter),
            "--randomize-lane-offset-min",
            str(args.randomize_lane_offset_min),
            "--randomize-lane-offset-max",
            str(args.randomize_lane_offset_max),
            "--carla-rpc-timeout",
            str(args.carla_rpc_timeout),
            "--cleanup-destroy-mode",
            args.cleanup_destroy_mode,
        ]
    )
    _append_flag(cmd, "--randomize-scenarios", args.randomize_scenarios)
    _append_flag(cmd, "--randomize-lane-offsets", args.randomize_lane_offsets)
    _append_flag(cmd, "--render", args.render)
    _append_flag(cmd, "--no-cuda", args.no_cuda)
    _append_flag(cmd, "--purge-existing-actors-on-reset", args.purge_existing_actors_on_reset)
    if args.rerun_existing:
        cmd.append("--rerun-existing")


def run_normal(args: argparse.Namespace) -> None:
    root, config = _target_root_and_config(args)
    script_path = _validate_target(root, "run_ego_natural_seed_eval.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--backend",
        args.backend,
        "--config",
        str(config),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--seed-count",
        str(args.seed_count),
        "--hdv-control-mode",
        args.hdv_control_mode,
        "--traffic-manager-port",
        str(args.traffic_manager_port),
        "--hdv-tm-speed-difference",
        str(args.hdv_tm_speed_difference),
        "--hdv-tm-distance-to-leading-vehicle",
        str(args.hdv_tm_distance_to_leading_vehicle),
    ]
    if args.ego_dir:
        cmd.extend(["--ego-dir", _path(args.ego_dir)])
    _append_repeat(cmd, "--ego-checkpoint", args.ego_checkpoint)
    if args.output_dir:
        cmd.extend(["--output-dir", _path(args.output_dir)])
    if args.hdv_action:
        cmd.extend(["--hdv-action", args.hdv_action])
    if args.hdv_model:
        cmd.extend(["--hdv-model", args.hdv_model])
    _append_flag(cmd, "--include-final", args.include_final)
    _append_flag(cmd, "--label-with-source", args.label_with_source)
    _append_flag(cmd, "--no-hdv-tm-auto-lane-change", args.no_hdv_tm_auto_lane_change)
    _append_common_scene_args(cmd, args)
    _run(cmd, root, _base_env(args, root), args.dry_run)


def _append_adv_args(cmd: list[str], args: argparse.Namespace, config: Path, adv_model_dir: Path, joint_round_log: Path) -> None:
    cmd.extend(
        [
            "--config",
            str(config),
            "--adv-model-dir",
            str(adv_model_dir),
            "--joint-round-log",
            str(joint_round_log),
            "--adv-round-last",
            str(args.adv_round_last),
            "--adv-sample-count",
            str(args.adv_sample_count),
            "--eval-episodes-per-adv",
            str(args.eval_episodes_per_adv),
            "--eval-seed-stride",
            str(args.eval_seed_stride),
            "--num-adv",
            str(args.num_adv),
        ]
    )
    _append_common_scene_args(cmd, args)


def run_adv(args: argparse.Namespace) -> None:
    root, config, adv_model_dir, joint_round_log = _shared_target_paths(args)
    script_path = _validate_target(root, "evaluate_finetuned_ego_against_adv.py")
    cmd = [sys.executable, str(script_path), "--backend", args.backend]
    if args.ego_checkpoint:
        cmd.extend(["--ego-checkpoint", _path(args.ego_checkpoint)])
    if args.output_dir:
        cmd.extend(["--output-dir", _path(args.output_dir)])
    if args.base_dir:
        cmd.extend(["--base-dir", _path(args.base_dir)])
    _append_repeat(cmd, "--ego-search-dir", args.ego_search_dir)
    _append_repeat(cmd, "--adv-search-dir", args.adv_search_dir)
    _append_adv_args(cmd, args, config, adv_model_dir, joint_round_log)
    _run(cmd, root, _base_env(args, root), args.dry_run)


def run_adv_sweep(args: argparse.Namespace) -> None:
    root, config, adv_model_dir, joint_round_log = _shared_target_paths(args)
    script_path = _validate_target(root, "run_ego_checkpoint_sweep_eval.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--ego-dir",
        _path(args.ego_dir),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
    ]
    _append_repeat(cmd, "--ego-checkpoint", args.ego_checkpoint)
    if args.output_dir:
        cmd.extend(["--output-dir", _path(args.output_dir)])
    _append_flag(cmd, "--include-final", args.include_final)
    _append_flag(cmd, "--label-with-source", args.label_with_source)
    _append_adv_args(cmd, args, config, adv_model_dir, joint_round_log)
    _run(cmd, root, _base_env(args, root), args.dry_run)


def _add_shared_root_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--carla-evolution-root", default=str(DEFAULT_CARLA_EVOLUTION_ROOT))
    parser.add_argument("--backend", choices=["mock", "carla"], default="carla")
    parser.add_argument("--config", default="configs/carla_0915.yaml")
    parser.add_argument("--dry-run", action="store_true")


def _add_shared_scene_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario-preset", default="slow_front_left_escape")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--eval-seed-base", type=int, default=DEFAULT_EVAL_SEED_BASE)
    parser.add_argument("--num-natural", type=int, default=DEFAULT_NUM_NATURAL)
    parser.add_argument("--spawn-start-index", type=int, default=0)
    parser.add_argument("--randomize-scenarios", action="store_true", default=True)
    parser.add_argument("--no-randomize-scenarios", dest="randomize_scenarios", action="store_false")
    parser.add_argument("--relative-offset-jitter", type=float, default=DEFAULT_RELATIVE_OFFSET_JITTER)
    parser.add_argument("--randomize-lane-offsets", action="store_true", default=True)
    parser.add_argument("--no-randomize-lane-offsets", dest="randomize_lane_offsets", action="store_false")
    parser.add_argument("--randomize-lane-offset-min", type=int, default=DEFAULT_LANE_OFFSET_MIN)
    parser.add_argument("--randomize-lane-offset-max", type=int, default=DEFAULT_LANE_OFFSET_MAX)
    parser.add_argument("--carla-rpc-timeout", type=float, default=300.0)
    parser.add_argument("--cleanup-destroy-mode", choices=["sequential", "batch"], default="sequential")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--purge-existing-actors-on-reset", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.set_defaults(randomize_scenarios=True, randomize_lane_offsets=True)


def _add_adv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adv-model-dir",
        default=f"results/{DEFAULT_ADV_RUN}/models/adv",
        help="Path to frozen MAPPO adversary checkpoints.",
    )
    parser.add_argument(
        "--joint-round-log",
        default=f"results/{DEFAULT_ADV_RUN}/joint_round_log.csv",
        help="Round log used to reproduce the sampled 10 adversary checkpoints.",
    )
    parser.add_argument("--adv-round-last", type=int, default=DEFAULT_ADV_ROUND_LAST)
    parser.add_argument("--adv-sample-count", type=int, default=DEFAULT_ADV_SAMPLE_COUNT)
    parser.add_argument("--eval-episodes-per-adv", type=int, default=10)
    parser.add_argument("--eval-seed-stride", type=int, default=DEFAULT_EVAL_SEED_STRIDE)
    parser.add_argument("--num-adv", type=int, default=DEFAULT_NUM_ADV)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    normal = subparsers.add_parser(
        "normal",
        help="Evaluate frozen ego policies with natural vehicles controlled by TrafficManager by default.",
    )
    _add_shared_root_args(normal)
    _add_shared_scene_args(normal)
    normal.add_argument("--ego-dir", default=None)
    normal.add_argument("--ego-checkpoint", action="append", default=None)
    normal.add_argument("--checkpoint-interval", type=int, default=999999999)
    normal.add_argument("--include-final", action="store_true")
    normal.add_argument("--label-with-source", action="store_true")
    normal.add_argument("--seed-count", type=int, default=10)
    normal.add_argument("--hdv-control-mode", choices=["traffic_manager", "fixed", "policy"], default="traffic_manager")
    normal.add_argument("--hdv-action", default="keep_lane")
    normal.add_argument("--hdv-model", default="")
    normal.add_argument("--traffic-manager-port", type=int, default=8000)
    normal.add_argument("--hdv-tm-speed-difference", type=float, default=0.0)
    normal.add_argument("--hdv-tm-distance-to-leading-vehicle", type=float, default=8.0)
    normal.add_argument("--no-hdv-tm-auto-lane-change", action="store_true")
    normal.add_argument(
        "--output-dir",
        default="results/ego_natural_seed_eval/scenge_normal_tm_hdv3_random10",
    )
    normal.set_defaults(func=run_normal)

    adv = subparsers.add_parser(
        "adv",
        help="Evaluate one frozen ego checkpoint against sampled frozen MAPPO adversaries.",
    )
    _add_shared_root_args(adv)
    _add_shared_scene_args(adv)
    _add_adv_args(adv)
    adv.set_defaults(num_natural=0)
    adv.add_argument("--ego-checkpoint", default=None)
    adv.add_argument("--ego-search-dir", action="append", default=None)
    adv.add_argument("--adv-search-dir", action="append", default=None)
    adv.add_argument("--base-dir", default=None)
    adv.add_argument(
        "--output-dir",
        default="results/ego_adv_frozen_eval/scenge_adv10_random_pos_lane",
    )
    adv.set_defaults(func=run_adv)

    sweep = subparsers.add_parser(
        "adv-sweep",
        help="Evaluate a directory or list of ego checkpoints against sampled frozen MAPPO adversaries.",
    )
    _add_shared_root_args(sweep)
    _add_shared_scene_args(sweep)
    _add_adv_args(sweep)
    sweep.set_defaults(num_natural=0)
    sweep.add_argument("--ego-dir", required=True)
    sweep.add_argument("--ego-checkpoint", action="append", default=None)
    sweep.add_argument("--checkpoint-interval", type=int, default=100)
    sweep.add_argument("--include-final", action="store_true")
    sweep.add_argument("--label-with-source", action="store_true")
    sweep.add_argument(
        "--output-dir",
        default="results/ego_adv_frozen_eval/scenge_adv10_sweep_random_pos_lane",
    )
    sweep.set_defaults(func=run_adv_sweep)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
