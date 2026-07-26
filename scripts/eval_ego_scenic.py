"""Visual evaluation for a PPO ego policy on generated ScenGE/Scenic scenarios."""

from __future__ import annotations

import argparse
from typing import Optional

import torch

from safebench.util.launch import (
    REPO_ROOT,
    LaunchArgs,
    prepare_run_kwargs,
    select_runner,
)


def _parse_route_id(value: Optional[str]):
    if value is None:
        return None
    if value.lower() in ("none", "all"):
        return None
    return [int(item) for item in value.split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_cfg", type=str, default="ppo.yaml")
    parser.add_argument("--scenario_cfg", type=str, default="train_scenario_scenic.yaml")
    parser.add_argument("--scenario-ids", type=int, nargs="+", default=[1, 3, 4])
    parser.add_argument("--eval-episodes", type=int, default=10)

    parser.add_argument("--ROOT_DIR", type=str, default=REPO_ROOT)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--frame_skip", "-fs", type=int, default=1)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm_port", type=int, default=8000)
    parser.add_argument("--fixed_delta_seconds", type=float, default=0.1)
    parser.add_argument("--max_episode_step", type=int, default=200)
    parser.add_argument("--auto_ego", action="store_true")
    parser.add_argument(
        "--route-id",
        type=str,
        default="1",
        help="Comma-separated scenic route ids, or all.",
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        default="safebench/agent/model_ckpt/safe_rl/ppo_scenge_mixed_ego/model_save",
        help="PPO checkpoint directory or checkpoint file.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    scenario_label = "_".join(str(item) for item in args.scenario_ids)
    launch = LaunchArgs(
        agent_cfg=args.agent_cfg,
        scenario_cfg=args.scenario_cfg,
        scenario_id=args.scenario_ids[0],
        mode="eval_ego",
        output_dir=f"log/eval_ego/ppo/scenario_mix_{scenario_label}",
        exp_name="exp",
        seed=args.seed,
        threads=args.threads,
        device=args.device,
        num_scenario=1,
        render=args.render,
        save_video=args.save_video,
        frame_skip=args.frame_skip,
        port=args.port,
        tm_port=args.tm_port,
        fixed_delta_seconds=args.fixed_delta_seconds,
        auto_ego=args.auto_ego,
        max_episode_step=args.max_episode_step,
        ROOT_DIR=args.ROOT_DIR,
    )
    agent_config, scenario_config = prepare_run_kwargs(launch)
    agent_config["load_dir"] = args.load_dir
    agent_config["pretrain_dir"] = None
    scenario_config["scenario_id"] = args.scenario_ids
    scenario_config["route_id"] = _parse_route_id(args.route_id)
    scenario_config["eval_episodes"] = args.eval_episodes

    Runner = select_runner(scenario_config)
    runner = Runner(agent_config, scenario_config)
    try:
        runner.run()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
