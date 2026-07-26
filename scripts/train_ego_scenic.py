"""Train the ego policy on generated ScenGE/Scenic scenarios.

This is PPO-only: the generated Scenic files define the scenario distribution,
while the ego vehicle is updated from CARLA environment rewards.
"""

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
    parser.add_argument("--scenario_id", type=int, default=None)
    parser.add_argument(
        "--scenario-ids",
        type=int,
        nargs="+",
        default=None,
        help="Train one PPO ego policy over multiple generated scenario ids.",
    )

    parser.add_argument("--ROOT_DIR", type=str, default=REPO_ROOT)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--num_scenario", "-ns", type=int, default=1)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--frame_skip", "-fs", type=int, default=1)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm_port", type=int, default=8000)
    parser.add_argument("--fixed_delta_seconds", type=float, default=0.1)
    parser.add_argument("--max_episode_step", type=int, default=300)
    parser.add_argument("--auto_ego", action="store_true")
    parser.add_argument("--traj_root", type=str, default=None)
    parser.add_argument("--record_traj", action="store_true")

    parser.add_argument(
        "--route-id",
        type=str,
        default=None,
        help="Comma-separated scenic route ids, or all. Default uses scenario yaml.",
    )
    parser.add_argument(
        "--sample-num",
        type=int,
        default=None,
        help="Override number of Scenic samples per behavior/route.",
    )
    parser.add_argument(
        "--train-episode",
        type=int,
        default=None,
        help="Override PPO training epochs from the agent yaml.",
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        default=None,
        help="Override PPO checkpoint output/load directory.",
    )
    parser.add_argument(
        "--pretrain-dir",
        type=str,
        default=None,
        help="Optional pretrained PPO .pt/.torch file, or null.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    scenario_ids = args.scenario_ids
    if scenario_ids is None:
        if args.scenario_id is None:
            raise ValueError("Provide either --scenario_id or --scenario-ids.")
        scenario_ids = [args.scenario_id]
    scenario_label = "_".join(str(item) for item in scenario_ids)
    launch_scenario_id = scenario_ids[0]

    launch = LaunchArgs(
        agent_cfg=args.agent_cfg,
        scenario_cfg=args.scenario_cfg,
        scenario_id=launch_scenario_id,
        mode="train_ego",
        output_dir="log/train_ego/ppo",
        exp_name=f"scenario_{scenario_label}",
        seed=args.seed,
        threads=args.threads,
        device=args.device,
        num_scenario=args.num_scenario,
        render=args.render,
        save_video=args.save_video,
        frame_skip=args.frame_skip,
        port=args.port,
        tm_port=args.tm_port,
        fixed_delta_seconds=args.fixed_delta_seconds,
        auto_ego=args.auto_ego,
        max_episode_step=args.max_episode_step,
        ROOT_DIR=args.ROOT_DIR,
        traj_root=args.traj_root,
        record_traj=args.record_traj,
    )
    agent_config, scenario_config = prepare_run_kwargs(launch)
    if len(scenario_ids) > 1:
        output_dir = f"log/train_ego/ppo/scenario_mix_{scenario_label}"
        agent_config["output_dir"] = output_dir
        scenario_config["output_dir"] = output_dir
        agent_config["exp_name"] = "exp"
        scenario_config["exp_name"] = "exp"
        scenario_config["scenario_id"] = scenario_ids
        scenario_config["mixed_training"] = True
    else:
        scenario_config["scenario_id"] = scenario_ids[0]
        scenario_config["mixed_training"] = False

    if args.route_id is not None:
        scenario_config["route_id"] = _parse_route_id(args.route_id)
    if args.sample_num is not None:
        scenario_config["sample_num"] = args.sample_num
    if args.train_episode is not None:
        agent_config["train_episode"] = args.train_episode
    if args.load_dir is not None:
        agent_config["load_dir"] = args.load_dir
    if args.pretrain_dir is not None:
        agent_config["pretrain_dir"] = None if args.pretrain_dir == "null" else args.pretrain_dir

    Runner = select_runner(scenario_config)
    runner = Runner(agent_config, scenario_config)
    try:
        runner.run()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
