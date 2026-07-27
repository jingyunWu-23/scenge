"""Run a CARLA Evolution script with SafeBench PPO as the ego adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import importlib
import runpy
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carla-evolution-root", required=True)
    parser.add_argument("--target-script", required=True)
    parser.add_argument("--ego-adapter", choices=["native", "safebench-ppo"], default="safebench-ppo")
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    carla_root = Path(args.carla_evolution_root).expanduser().resolve()
    target_script = Path(args.target_script).expanduser().resolve()
    for path in (
        str(REPO_ROOT),
        str(REPO_ROOT / "MARL1"),
        str(carla_root),
        str(carla_root / "scripts"),
        str(carla_root.parent / "MARL1"),
        str(carla_root / "MARL1"),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    if "carla_evolution" not in sys.modules:
        package = types.ModuleType("carla_evolution")
        package.__path__ = [str(carla_root)]
        sys.modules["carla_evolution"] = package

    finetune = importlib.import_module("carla_evolution.scripts.finetune_ego_enhanced_poet")
    sys.modules.setdefault("finetune_ego_enhanced_poet", finetune)

    if args.ego_adapter == "safebench-ppo":
        from safebench.scenge.carla_evolution_ego_adapter import SafeBenchPPOToDiscreteAdapter

        ego_ppo_module = types.ModuleType("carla_evolution.agents.ego_ppo")
        ego_ppo_module.EgoPPOAdapter = SafeBenchPPOToDiscreteAdapter
        sys.modules["carla_evolution.agents.ego_ppo"] = ego_ppo_module

        def patched_runtime_imports():
            from carla_evolution.agents.mappo import MAPPOAgent
            from carla_evolution.envs.factory import make_env
            from carla_evolution.training import train as train_mod

            return train_mod, MAPPOAgent, SafeBenchPPOToDiscreteAdapter, make_env

        finetune.runtime_imports = patched_runtime_imports

    target_args = list(args.target_args)
    if target_args and target_args[0] == "--":
        target_args = target_args[1:]
    sys.argv = [str(target_script), *target_args]
    runpy.run_path(str(target_script), run_name="__main__")


if __name__ == "__main__":
    main()
