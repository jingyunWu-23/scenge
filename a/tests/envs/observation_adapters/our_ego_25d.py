"""Current-project 25-D ego observation adapter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    from envs.observation import LowDimObservationBuilder
except ImportError:  # pragma: no cover - allows package-style execution.
    from carla_evolution.envs.observation import LowDimObservationBuilder

from .base import ObservationAdapter


class OurEgo25DObservationAdapter(ObservationAdapter):
    """Build the same 25-D observation used by the current EgoPPO policy."""

    name = "our_ego_25d"

    def __init__(self, state_dim: int = 25, lane_width: float = 4.0, max_lanes: int = 4):
        self.builder = LowDimObservationBuilder(
            state_dim=int(state_dim),
            lane_width=float(lane_width),
            max_lanes=int(max_lanes),
        )

    def build(self, scene_state: Any, ego_vehicle: Optional[Any] = None, route: Any = None) -> np.ndarray:
        del route
        vehicle = ego_vehicle if ego_vehicle is not None else getattr(scene_state, "ego_vehicle", None)
        if vehicle is None:
            return np.zeros(self.builder.state_dim, dtype=np.float32)
        sync = getattr(scene_state, "sync_route_progress_from_ego", None)
        if callable(sync):
            sync()
        return self.builder.build_vehicle_observation(scene_state, vehicle).astype(np.float32)

    def build_ego(self, scene_state: Any) -> np.ndarray:
        return self.build(scene_state)
