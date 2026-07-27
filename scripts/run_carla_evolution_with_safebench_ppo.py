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
    for path in (str(REPO_ROOT), str(carla_root), str(carla_root / "scripts")):
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

        original_runtime_imports = finetune.runtime_imports

        def patched_runtime_imports():
            train_mod, MAPPOAgent, _EgoPPOAdapter, make_env = original_runtime_imports()
            return train_mod, MAPPOAgent, SafeBenchPPOToDiscreteAdapter, make_env

        finetune.runtime_imports = patched_runtime_imports

    target_args = list(args.target_args)
    if target_args and target_args[0] == "--":
        target_args = target_args[1:]
    sys.argv = [str(target_script), *target_args]
    runpy.run_path(str(target_script), run_name="__main__")


if __name__ == "__main__":
    main()
