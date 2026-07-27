"""Environment factory helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .env import EvolutionEnv
from .legacy_adapter import LegacyTrainingAdapter


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(str(config_path))
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to load YAML config files") from exc
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _flatten_config(raw)


def make_env(config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None, legacy: bool = False):
    merged = load_config(config_path) if config_path else {}
    if config:
        merged.update(config)
    env = EvolutionEnv(merged)
    return LegacyTrainingAdapter(env) if legacy else env


def _flatten_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for value in raw.values():
        if isinstance(value, dict):
            flat.update(value)
    flat.update({key: value for key, value in raw.items() if not isinstance(value, dict)})
    return flat
