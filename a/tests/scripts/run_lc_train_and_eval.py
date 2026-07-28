"""Run LC straight-follow training and evaluation as one experiment command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Train LC straight-follow scenes and evaluate ego policies.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-scenes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--route-length", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=669)
    parser.add_argument("--target-speed", type=float, default=18.0)
    parser.add_argument("--ego-checkpoints", default="", help="Comma-separated EgoPPO checkpoints for evaluation.")
    parser.add_argument("--ego-action", choices=["keep_lane", "accelerate", "decelerate"], default="accelerate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-weight", type=float, default=1e-4)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-train", action="store_true", default=False)
    parser.add_argument("--skip-eval", action="store_true", default=False)
    parser.add_argument("--scene-params-csv", default=None, help="Use existing fixed scenes when --skip-train is set.")
    parser.add_argument("--lc-checkpoint", default=None, help="Use existing LC checkpoint when --skip-train is set.")
    return parser.parse_args()


def build_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%b_%d_%H_%M_%S")
        output_dir = MODEL_ROOT / "tests" / "results" / f"lc_straight_full_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_command(cmd):
    print(" ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    output_dir = build_output_dir(args)
    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"

    scene_params_csv = Path(args.scene_params_csv) if args.scene_params_csv else train_dir / "decoded_scene_params.csv"
    lc_checkpoint = Path(args.lc_checkpoint) if args.lc_checkpoint else train_dir / "models" / f"0_epoch{args.episodes}.pt"

    if not args.skip_train:
        train_cmd = [
            sys.executable,
            str(MODEL_ROOT / "tests" / "scripts" / "train_lc_straight_templates.py"),
            "--episodes",
            str(args.episodes),
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
            "--ego-action",
            str(args.ego_action),
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
            str(train_dir),
        ]
        if args.no_cuda:
            train_cmd.append("--no-cuda")
        run_command(train_cmd)

    if not args.skip_eval:
        eval_cmd = [
            sys.executable,
            str(MODEL_ROOT / "tests" / "scripts" / "eval_lc_straight_templates.py"),
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
            "--ego-action",
            str(args.ego_action),
            "--output-dir",
            str(eval_dir),
        ]
        if scene_params_csv.exists():
            eval_cmd.extend(["--scene-params-csv", str(scene_params_csv)])
        else:
            eval_cmd.extend(["--lc-checkpoint", str(lc_checkpoint), "--num-scenes", str(args.eval_scenes)])
        if args.ego_checkpoints:
            eval_cmd.extend(["--ego-checkpoints", args.ego_checkpoints])
        if args.no_cuda:
            eval_cmd.append("--no-cuda")
        run_command(eval_cmd)

    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
