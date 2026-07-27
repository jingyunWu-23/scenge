"""Observation adapter interfaces for LC straight-road tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ObservationAdapter(ABC):
    name = "base"

    @abstractmethod
    def build(self, scene_state: Any, ego_vehicle: Any = None, route: Any = None) -> np.ndarray:
        """Build an observation for one ego-policy method."""
        raise NotImplementedError
