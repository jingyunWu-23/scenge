"""Straight-road LC test templates."""

from .cut_in import CutInTemplate
from .lane_change import LaneChangeTemplate
from .passing import PassingTemplate
from .straight_follow import FOLLOW_BEHAVIORS, StraightFollowTemplate

TEMPLATE_REGISTRY = {
    StraightFollowTemplate.name: StraightFollowTemplate,
    PassingTemplate.name: PassingTemplate,
    LaneChangeTemplate.name: LaneChangeTemplate,
    CutInTemplate.name: CutInTemplate,
}


def make_template(template_name: str, **kwargs):
    try:
        template_cls = TEMPLATE_REGISTRY[str(template_name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported LC template: {template_name}") from exc
    return template_cls(**kwargs)


__all__ = [
    "CutInTemplate",
    "FOLLOW_BEHAVIORS",
    "LaneChangeTemplate",
    "PassingTemplate",
    "StraightFollowTemplate",
    "TEMPLATE_REGISTRY",
    "make_template",
]
