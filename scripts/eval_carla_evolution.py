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
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARLA_EVOLUTION_ROOT = REPO_ROOT / "a"
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


def _output_dir_arg(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def _append_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def _append_repeat(cmd: list[str], flag: str, values: Sequence[str] | None) -> None:
    for value in values or ():
        cmd.extend([flag, value])


def _adapt_checkpoint_path(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Ego checkpoint not found: {source}")
    if source.suffix != ".torch":
        return str(source)
    cache_dir = REPO_ROOT / ".cache" / "carla_evolution_ego_adapter"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = int(hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12], 16)
    target = cache_dir / f"checkpoint-{digest}.pt"
    _copy_checkpoint(source, target)
    return str(target)


def _adapt_checkpoint_dir(path: str | Path) -> str:
    source_dir = Path(path).expanduser().resolve()
    if list(source_dir.glob("checkpoint-*.pt")):
        return str(source_dir)
    torch_files = sorted(source_dir.glob("*.torch"))
    if not torch_files:
        return str(source_dir)
    cache_dir = REPO_ROOT / ".cache" / "carla_evolution_ego_adapter" / source_dir.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    _remove_checkpoint_symlinks(cache_dir)
    for index, source in enumerate(torch_files):
        digest = int(hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8], 16)
        step = index * 100000 + digest % 100000
        _copy_checkpoint(source, cache_dir / f"checkpoint-{step}.pt")
    return str(cache_dir)


def _checkpoint_dir_from_checkpoints(paths: Sequence[str] | None) -> str | None:
    checkpoints = [Path(path).expanduser().resolve() for path in (paths or ())]
    torch_files = [path for path in checkpoints if path.suffix == ".torch"]
    if not torch_files:
        return None
    cache_dir = REPO_ROOT / ".cache" / "carla_evolution_ego_adapter" / "explicit_checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _clear_checkpoint_files(cache_dir)
    for source in torch_files:
        if not source.is_file():
            raise FileNotFoundError(f"Ego checkpoint not found: {source}")
        digest = int(hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12], 16)
        _copy_checkpoint(source, cache_dir / f"checkpoint-{digest}.pt")
    return str(cache_dir)


def _copy_checkpoint(source: Path, target: Path) -> None:
    if target.is_symlink():
        target.unlink()
    if target.exists():
        same_size = target.stat().st_size == source.stat().st_size
        same_mtime = int(target.stat().st_mtime) == int(source.stat().st_mtime)
        if same_size and same_mtime:
            return
        target.unlink()
    shutil.copy2(source, target)


def _remove_checkpoint_symlinks(directory: Path) -> None:
    for path in directory.glob("checkpoint-*.pt"):
        if path.is_symlink():
            path.unlink()


def _clear_checkpoint_files(directory: Path) -> None:
    for path in directory.glob("checkpoint-*.pt"):
        path.unlink()


def _checkpoint_arg(args: argparse.Namespace, path: str | Path) -> str:
    if args.ego_adapter in {"safebench-ppo", "chatscene-ppo"}:
        return _adapt_checkpoint_path(path)
    return _path(path)


def _checkpoint_dir_arg(args: argparse.Namespace, path: str | Path) -> str:
    if args.ego_adapter in {"safebench-ppo", "chatscene-ppo"}:
        return _adapt_checkpoint_dir(path)
    return _path(path)


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
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env.setdefault(key, "1")
    pythonpath = [str(root), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if getattr(args, "cuda_visible_devices", None):
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["SCENGE_ROOT"] = str(REPO_ROOT)
    env["SAFEBENCH_EGO_AGENT_CFG"] = str(args.safebench_agent_cfg)
    if args.safebench_obs_indices:
        env["SAFEBENCH_EGO_OBS_INDICES"] = str(args.safebench_obs_indices)
    env["SAFEBENCH_EGO_STEER_THRESHOLD"] = str(args.safebench_steer_threshold)
    env["SAFEBENCH_EGO_ACC_THRESHOLD"] = str(args.safebench_acc_threshold)
    return env


def _target_command(args: argparse.Namespace, root: Path, script_path: Path) -> list[str]:
    if args.ego_adapter == "native" and root.name == "carla_evolution":
        return [sys.executable, str(script_path)]
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_carla_evolution_with_safebench_ppo.py"),
        "--carla-evolution-root",
        str(root),
        "--target-script",
        str(script_path),
        "--ego-adapter",
        args.ego_adapter,
        "--",
    ]


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
        *_target_command(args, root, script_path),
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
    ego_dir = _checkpoint_dir_arg(args, args.ego_dir) if args.ego_dir else None
    if args.ego_adapter in {"safebench-ppo", "chatscene-ppo"} and not ego_dir:
        ego_dir = _checkpoint_dir_from_checkpoints(args.ego_checkpoint)
    if ego_dir:
        cmd.extend(["--ego-dir", ego_dir])
    if args.ego_adapter not in {"safebench-ppo", "chatscene-ppo"}:
        for checkpoint in args.ego_checkpoint or ():
            cmd.extend(["--ego-checkpoint", _checkpoint_arg(args, checkpoint)])
    if args.output_dir:
        cmd.extend(["--output-dir", _output_dir_arg(args.output_dir)])
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
    cmd = [*_target_command(args, root, script_path), "--backend", args.backend]
    if args.ego_checkpoint:
        cmd.extend(["--ego-checkpoint", _checkpoint_arg(args, args.ego_checkpoint)])
    if args.output_dir:
        cmd.extend(["--output-dir", _output_dir_arg(args.output_dir)])
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
        *_target_command(args, root, script_path),
        "--ego-dir",
        _checkpoint_dir_arg(args, args.ego_dir),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
    ]
    for checkpoint in args.ego_checkpoint or ():
        cmd.extend(["--ego-checkpoint", _checkpoint_arg(args, checkpoint)])
    if args.output_dir:
        cmd.extend(["--output-dir", _output_dir_arg(args.output_dir)])
    _append_flag(cmd, "--include-final", args.include_final)
    _append_flag(cmd, "--label-with-source", args.label_with_source)
    _append_adv_args(cmd, args, config, adv_model_dir, joint_round_log)
    _run(cmd, root, _base_env(args, root), args.dry_run)


def run_safebench_lc(args: argparse.Namespace) -> None:
    root, config = _target_root_and_config(args)
    script_path = root / "tests" / "scripts" / "run_safebench_inspired_carla_ego_sweep.py"
    adv_model_dir = Path(args.adv_model_dir).expanduser()
    if not adv_model_dir.is_absolute():
        adv_model_dir = root / adv_model_dir
    missing = [
        path
        for path in (
            root,
            script_path,
            root / "tests" / "configs" / "advsim.json",
            root / "scripts" / "finetune_ego_enhanced_poet.py",
            root / "scripts" / "evaluate_ego_adv.py",
            root / "envs" / "env.py",
            root / "agents" / "mappo.py",
        )
        if not path.exists()
    ]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise SystemExit(f"Missing required SafeBench LC test files:\n  {formatted}")

    scenario_type_json = Path(args.scenario_type_json).expanduser()
    if not scenario_type_json.is_absolute():
        scenario_type_json = root / scenario_type_json
    ego_checkpoint = _checkpoint_arg(args, args.ego_checkpoint)
    cmd = [
        *_target_command(args, root, script_path),
        "--training-ego-checkpoint",
        ego_checkpoint,
        "--training-ego-label",
        args.model_label,
        "--finetune-ego-checkpoints",
        "none",
        "--config",
        str(config),
        "--scenario-type-json",
        str(scenario_type_json),
        "--scenario-ids",
        args.scenario_ids,
        "--route-ids",
        args.route_ids,
        "--max-routes-per-scenario",
        str(args.max_routes_per_scenario),
        "--max-inits-per-route",
        str(args.max_inits_per_route),
        "--adv-model-dir",
        str(adv_model_dir),
        "--adv-step",
        str(args.adv_step),
        "--num-adv",
        str(args.num_adv),
        "--num-natural",
        str(args.num_natural),
        "--hdv-action",
        args.hdv_action,
        "--max-steps",
        str(args.max_steps),
        "--route-length",
        str(args.route_length),
        "--target-speed-cruise",
        str(args.target_speed_cruise),
        "--seed",
        str(args.seed),
        "--torch-seed",
        str(args.torch_seed),
        "--carla-rpc-timeout",
        str(args.carla_rpc_timeout),
        "--cleanup-destroy-mode",
        args.cleanup_destroy_mode,
        "--case-cooldown",
        str(args.case_cooldown),
        "--output-dir",
        _output_dir_arg(args.output_dir),
    ]
    _append_flag(cmd, "--no-cuda", args.no_cuda)
    _append_flag(cmd, "--render", args.render)
    _append_flag(cmd, "--purge-existing-actors-on-reset", args.purge_existing_actors_on_reset)
    _append_flag(cmd, "--dry-run", args.dry_run_lc)
    _run(cmd, root, _base_env(args, root), args.dry_run)


def run_lc_eval(args: argparse.Namespace) -> None:
    root, _ = _target_root_and_config(args)
    script_path = root / "tests" / "scripts" / "eval_lc_straight_templates.py"
    missing = [
        path
        for path in (
            root,
            script_path,
            root / "tests" / "envs" / "scene_state.py",
            root / "tests" / "scenarios" / "templates" / "straight_follow.py",
        )
        if not path.exists()
    ]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise SystemExit(f"Missing required LC eval files:\n  {formatted}")
    scene_params_csv = Path(args.scene_params_csv).expanduser()
    if not scene_params_csv.is_absolute():
        scene_params_csv = root / scene_params_csv
    cmd = [
        *_target_command(args, root, script_path),
        "--scene-params-csv",
        str(scene_params_csv),
        "--ego-checkpoints",
        _checkpoint_arg(args, args.ego_checkpoint),
        "--ego-labels",
        args.model_label,
        "--max-steps",
        str(args.max_steps),
        "--route-length",
        str(args.route_length),
        "--dt",
        str(args.dt),
        "--seed",
        str(args.seed),
        "--target-speed",
        str(args.target_speed),
        "--output-dir",
        _output_dir_arg(args.output_dir),
    ]
    _append_flag(cmd, "--no-cuda", args.no_cuda)
    _run(cmd, root, _base_env(args, root), args.dry_run)


def run_lc_generate_scenes(args: argparse.Namespace) -> None:
    root, _ = _target_root_and_config(args)
    script_path = root / "tests" / "scripts" / "train_lc_straight_templates.py"
    missing = [
        path
        for path in (
            root,
            script_path,
            root / "tests" / "envs" / "scene_state.py",
            root / "tests" / "scenarios" / "parameter_space.py",
        )
        if not path.exists()
    ]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise SystemExit(f"Missing required LC generation files:\n  {formatted}")
    output_dir = Path(_output_dir_arg(args.output_dir))
    template_names = [item.strip() for item in str(args.template_names).split(",") if item.strip()]
    generated_csvs = []
    for index, template_name in enumerate(template_names):
        template_dir = output_dir / template_name / "train"
        cmd = [
            sys.executable,
            str(script_path),
            "--template-name",
            template_name,
            "--episodes",
            str(args.episodes_per_template),
            "--max-steps",
            str(args.max_steps),
            "--route-length",
            str(args.route_length),
            "--dt",
            str(args.dt),
            "--seed",
            str(int(args.seed) + index * 100000),
            "--target-speed",
            str(args.target_speed),
            "--ego-action",
            args.ego_action,
            "--batch-size",
            str(args.batch_size),
            "--buffer-capacity",
            str(args.buffer_capacity),
            "--lr",
            str(args.lr),
            "--entropy-weight",
            str(args.entropy_weight),
            "--save-interval",
            str(args.save_interval),
            "--output-dir",
            str(template_dir),
        ]
        _append_flag(cmd, "--no-cuda", args.no_cuda)
        _run(cmd, root, _base_env(args, root), args.dry_run)
        generated_csvs.append(template_dir / "decoded_scene_params.csv")
    if not args.dry_run:
        combined_csv = output_dir / "decoded_scene_params.csv"
        _combine_lc_scene_csvs(generated_csvs, combined_csv)
        print(f"Combined LC scene params: {combined_csv}", flush=True)


def _combine_lc_scene_csvs(source_paths: Sequence[Path], output_path: Path) -> None:
    import csv

    rows = []
    fieldnames = []
    for source_path in source_paths:
        with source_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for field in reader.fieldnames or ():
                if field not in fieldnames:
                    fieldnames.append(field)
            for row in reader:
                rows.append(row)
    if "episode" not in fieldnames:
        fieldnames.insert(0, "episode")
    for episode, row in enumerate(rows, start=1):
        row["episode"] = str(episode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _add_shared_root_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--carla-evolution-root", default=str(DEFAULT_CARLA_EVOLUTION_ROOT))
    parser.add_argument("--backend", choices=["mock", "carla"], default="carla")
    parser.add_argument("--config", default="configs/carla_0915.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ego-adapter", choices=["native", "safebench-ppo", "chatscene-ppo"], default="native")
    parser.add_argument("--safebench-agent-cfg", default="ppo.yaml")
    parser.add_argument(
        "--safebench-obs-indices",
        default="",
        help="Comma-separated indices from CARLA Evolution ego obs to feed SafeBench PPO. Default: first 4 values.",
    )
    parser.add_argument("--safebench-steer-threshold", type=float, default=0.1)
    parser.add_argument("--safebench-acc-threshold", type=float, default=0.5)


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

    safebench_lc = subparsers.add_parser(
        "safebench-lc",
        help="Evaluate one ego checkpoint on the four straight-road LC templates: follow, passing, lane_change, cut_in.",
    )
    _add_shared_root_args(safebench_lc)
    safebench_lc.set_defaults(ego_adapter="safebench-ppo")
    safebench_lc.add_argument("--ego-checkpoint", required=True)
    safebench_lc.add_argument("--model-label", default="SafeBench-PPO")
    safebench_lc.add_argument("--scenario-type-json", default="tests/configs/advsim.json")
    safebench_lc.add_argument("--scenario-ids", default="1,2,3,4")
    safebench_lc.add_argument("--route-ids", default="")
    safebench_lc.add_argument("--max-routes-per-scenario", type=int, default=10)
    safebench_lc.add_argument("--max-inits-per-route", type=int, default=10)
    safebench_lc.add_argument("--adv-model-dir", default=f"results/{DEFAULT_ADV_RUN}/models/adv")
    safebench_lc.add_argument("--adv-step", type=int, default=0)
    safebench_lc.add_argument("--num-adv", type=int, default=DEFAULT_NUM_ADV)
    safebench_lc.add_argument("--num-natural", type=int, default=0)
    safebench_lc.add_argument("--hdv-action", default="keep_lane")
    safebench_lc.add_argument("--max-steps", type=int, default=200)
    safebench_lc.add_argument("--route-length", type=float, default=200.0)
    safebench_lc.add_argument("--target-speed-cruise", type=float, default=20.0)
    safebench_lc.add_argument("--seed", type=int, default=669)
    safebench_lc.add_argument("--torch-seed", type=int, default=669)
    safebench_lc.add_argument("--carla-rpc-timeout", type=float, default=300.0)
    safebench_lc.add_argument("--cleanup-destroy-mode", choices=["sequential", "batch"], default="sequential")
    safebench_lc.add_argument("--case-cooldown", type=float, default=0.2)
    safebench_lc.add_argument("--render", action="store_true")
    safebench_lc.add_argument("--no-cuda", action="store_true")
    safebench_lc.add_argument("--purge-existing-actors-on-reset", action="store_true")
    safebench_lc.add_argument("--dry-run-lc", action="store_true")
    safebench_lc.add_argument(
        "--output-dir",
        default="results/safebench_inspired_carla_4scenes/ppo_torch",
    )
    safebench_lc.set_defaults(func=run_safebench_lc)

    lc_eval = subparsers.add_parser(
        "lc-eval",
        help="Evaluate an ego checkpoint on scenes generated by the LC policy; no MAPPO adversary is loaded.",
    )
    _add_shared_root_args(lc_eval)
    lc_eval.set_defaults(ego_adapter="safebench-ppo")
    lc_eval.add_argument("--ego-checkpoint", required=True)
    lc_eval.add_argument("--model-label", default="ppo_torch")
    lc_eval.add_argument(
        "--scene-params-csv",
        required=True,
        help="LC-generated decoded_scene_params.csv from tests/scripts/train_lc_straight_templates.py.",
    )
    lc_eval.add_argument("--max-steps", type=int, default=200)
    lc_eval.add_argument("--route-length", type=float, default=200.0)
    lc_eval.add_argument("--dt", type=float, default=0.05)
    lc_eval.add_argument("--seed", type=int, default=669)
    lc_eval.add_argument("--target-speed", type=float, default=18.0)
    lc_eval.add_argument("--no-cuda", action="store_true")
    lc_eval.add_argument(
        "--output-dir",
        default="results/lc_generated_scene_eval/ppo_torch",
    )
    lc_eval.set_defaults(func=run_lc_eval)

    lc_generate = subparsers.add_parser(
        "lc-generate-scenes",
        help="Train/sample LC policies for the four straight-road templates and write a combined decoded_scene_params.csv.",
    )
    _add_shared_root_args(lc_generate)
    lc_generate.set_defaults(ego_adapter="native")
    lc_generate.add_argument("--template-names", default="straight_follow,passing,lane_change,cut_in")
    lc_generate.add_argument("--episodes-per-template", type=int, default=100)
    lc_generate.add_argument("--max-steps", type=int, default=200)
    lc_generate.add_argument("--route-length", type=float, default=200.0)
    lc_generate.add_argument("--dt", type=float, default=0.05)
    lc_generate.add_argument("--seed", type=int, default=669)
    lc_generate.add_argument("--target-speed", type=float, default=18.0)
    lc_generate.add_argument("--ego-action", choices=["keep_lane", "accelerate", "decelerate"], default="accelerate")
    lc_generate.add_argument("--batch-size", type=int, default=32)
    lc_generate.add_argument("--buffer-capacity", type=int, default=1000)
    lc_generate.add_argument("--lr", type=float, default=1e-3)
    lc_generate.add_argument("--entropy-weight", type=float, default=1e-4)
    lc_generate.add_argument("--save-interval", type=int, default=50)
    lc_generate.add_argument("--no-cuda", action="store_true")
    lc_generate.add_argument("--output-dir", default="results/lc_generated_scenes/4templates_ep100")
    lc_generate.set_defaults(func=run_lc_generate_scenes)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
