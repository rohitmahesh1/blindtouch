"""Object-family and stable-pose definitions for BlindTouch episodes.

The policy never receives these values.  They define the hidden physical
distribution and evaluation metadata used by the MuJoCo environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


Quaternion = tuple[float, float, float, float]
FloatRange = tuple[float, float]
SUPPORTED_GEOMS = ("cylinder", "box", "capsule", "ellipsoid", "chassis")
TOUCH_SKILL_MODES = (
    "round_retention",
    "rigid_asymmetric",
    "slippery_retention",
    "fragile_balance",
)


@dataclass(frozen=True)
class StablePose:
    """A physically stable initial orientation and its vertical support axis."""

    identifier: str
    quaternion: Quaternion
    resting_axis: str
    yaw_range: FloatRange = (-np.pi, np.pi)
    minimum_grasp_height: float = 0.020

    def resting_half_height(self, dimensions: tuple[float, float, float]) -> float:
        return dimensions[{"x": 0, "y": 1, "z": 2}[self.resting_axis]]

    def target_grasp_height(self, dimensions: tuple[float, float, float]) -> float:
        return max(self.resting_half_height(dimensions), self.minimum_grasp_height)


@dataclass(frozen=True)
class ObjectFamily:
    """Randomizable collision archetype used for training or evaluation."""

    identifier: str
    collision_kinds: tuple[str, ...]
    half_size_x_range: FloatRange
    half_size_y_range: FloatRange
    half_size_z_range: FloatRange
    mass_range: FloatRange
    friction_range: FloatRange
    safe_force_range: FloatRange
    visual_style: str
    stable_poses: tuple[StablePose, ...]
    skill_modes: tuple[str, ...]
    evaluation_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EpisodeObject:
    """One fully specified hidden object instance consumed by the environment."""

    family: str
    name: str
    shape: str
    pose: str
    quaternion: Quaternion
    half_size_x: float
    half_size_y: float
    half_size_z: float
    mass: float
    friction: float
    safe_force: float
    x_offset: float
    y_offset: float
    yaw: float
    resting_half_height: float
    grasp_height: float
    visual_style: str
    evaluation_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SamplingConfig:
    """Training distribution controls shared by training and evaluation code."""

    training_families: tuple[str, ...] = (
        "rounded",
        "container",
        "package",
        "slippery",
        "fragile",
        "chassis",
    )
    allowed_poses: tuple[str, ...] | None = None
    offset_range: FloatRange = (-0.006, 0.006)
    friction_range: FloatRange | None = None
    mass_range: FloatRange | None = None
    safe_force_headroom_range: FloatRange | None = None
    nominal_pad_force_capacity: float = 0.70
    holding_force_margin: float = 0.80
    safe_force_margin: float = 1.35


UPRIGHT = StablePose("upright", (1.0, 0.0, 0.0, 0.0), "z")
SIDE_X = StablePose(
    "side_x",
    (float(np.cos(np.pi / 4.0)), 0.0, float(np.sin(np.pi / 4.0)), 0.0),
    "x",
)
SIDE_Y = StablePose(
    "side_y",
    (float(np.cos(np.pi / 4.0)), float(np.sin(np.pi / 4.0)), 0.0, 0.0),
    "y",
)
WHEELS_DOWN = StablePose("wheels_down", (1.0, 0.0, 0.0, 0.0), "z", (-0.35, 0.35))


TRAINING_FAMILIES: dict[str, ObjectFamily] = {
    "rounded": ObjectFamily(
        identifier="rounded",
        collision_kinds=("ellipsoid",),
        half_size_x_range=(0.023, 0.031),
        half_size_y_range=(0.023, 0.031),
        half_size_z_range=(0.022, 0.040),
        mass_range=(0.030, 0.160),
        friction_range=(0.20, 1.10),
        safe_force_range=(0.40, 2.80),
        visual_style="abstract_rounded",
        stable_poses=(UPRIGHT,),
        skill_modes=("round_retention",),
        evaluation_tags=("training", "rounded", "round_retention"),
    ),
    "container": ObjectFamily(
        identifier="container",
        collision_kinds=("cylinder", "capsule"),
        half_size_x_range=(0.023, 0.030),
        half_size_y_range=(0.023, 0.030),
        half_size_z_range=(0.024, 0.042),
        mass_range=(0.040, 0.220),
        friction_range=(0.20, 1.10),
        safe_force_range=(0.40, 2.80),
        visual_style="abstract_container",
        stable_poses=(UPRIGHT, SIDE_X),
        skill_modes=("round_retention", "slippery_retention"),
        evaluation_tags=("training", "container", "round_retention"),
    ),
    "package": ObjectFamily(
        identifier="package",
        collision_kinds=("box",),
        half_size_x_range=(0.023, 0.032),
        half_size_y_range=(0.023, 0.032),
        half_size_z_range=(0.018, 0.040),
        mass_range=(0.030, 0.180),
        friction_range=(0.30, 1.20),
        safe_force_range=(0.40, 2.80),
        visual_style="abstract_package",
        stable_poses=(UPRIGHT, SIDE_X, SIDE_Y),
        skill_modes=("rigid_asymmetric",),
        evaluation_tags=("training", "package", "rigid_asymmetric"),
    ),
    "slippery": ObjectFamily(
        identifier="slippery",
        collision_kinds=("box", "cylinder", "capsule", "ellipsoid"),
        half_size_x_range=(0.021, 0.035),
        half_size_y_range=(0.020, 0.034),
        half_size_z_range=(0.014, 0.042),
        mass_range=(0.030, 0.150),
        friction_range=(0.16, 0.36),
        safe_force_range=(0.70, 3.00),
        visual_style="abstract_slippery",
        stable_poses=(UPRIGHT, SIDE_X, SIDE_Y),
        skill_modes=("slippery_retention",),
        evaluation_tags=("training", "slippery", "slippery_retention"),
    ),
    "fragile": ObjectFamily(
        identifier="fragile",
        collision_kinds=("ellipsoid", "box", "cylinder"),
        half_size_x_range=(0.022, 0.033),
        half_size_y_range=(0.021, 0.033),
        half_size_z_range=(0.018, 0.038),
        mass_range=(0.025, 0.090),
        friction_range=(0.65, 1.20),
        safe_force_range=(0.28, 0.90),
        visual_style="abstract_fragile",
        stable_poses=(UPRIGHT, SIDE_X, SIDE_Y),
        skill_modes=("fragile_balance",),
        evaluation_tags=("training", "fragile", "fragile_balance"),
    ),
    "chassis": ObjectFamily(
        identifier="chassis",
        collision_kinds=("chassis",),
        half_size_x_range=(0.032, 0.040),
        half_size_y_range=(0.020, 0.026),
        half_size_z_range=(0.014, 0.019),
        mass_range=(0.050, 0.180),
        friction_range=(0.30, 1.00),
        safe_force_range=(0.70, 2.80),
        visual_style="abstract_chassis",
        stable_poses=(WHEELS_DOWN,),
        skill_modes=("rigid_asymmetric",),
        evaluation_tags=("training", "compound", "rigid_asymmetric"),
    )
}


def sample_training_object(
    rng: np.random.Generator, config: SamplingConfig | None = None
) -> EpisodeObject:
    """Sample one hidden training object from physical archetypes."""

    settings = config or SamplingConfig()
    family_name = str(rng.choice(settings.training_families))
    if family_name not in TRAINING_FAMILIES:
        raise ValueError(f"Unsupported active training family: {family_name!r}")
    family = TRAINING_FAMILIES[family_name]
    for mode in family.skill_modes:
        if mode not in TOUCH_SKILL_MODES:
            raise ValueError(f"Family {family.identifier!r} declares unknown skill mode {mode!r}")
    shape = str(rng.choice(family.collision_kinds))
    stable_poses = (
        family.stable_poses
        if settings.allowed_poses is None
        else tuple(pose for pose in family.stable_poses if pose.identifier in settings.allowed_poses)
    )
    if not stable_poses:
        raise ValueError(
            f"Active family {family_name!r} has no pose in allowed_poses={settings.allowed_poses!r}"
        )
    pose = stable_poses[int(rng.integers(len(stable_poses)))]
    x, y, z = _sample_dimensions(rng, family, shape)
    friction_range = _intersect_range(family.friction_range, settings.friction_range, "friction")
    mass_range = _intersect_range(family.mass_range, settings.mass_range, "mass")
    friction = float(rng.uniform(*friction_range))
    feasible_mass_ceiling = (
        3.0
        * friction
        * settings.nominal_pad_force_capacity
        * settings.holding_force_margin
        / 9.81
    )
    upper_mass = max(mass_range[0], min(mass_range[1], feasible_mass_ceiling))
    mass = float(rng.uniform(mass_range[0], upper_mass))
    required_pad_force = mass * 9.81 / (3.0 * friction)
    safe_force = _sample_safe_force(rng, family, settings, required_pad_force)
    yaw = float(rng.uniform(*pose.yaw_range))
    dimensions = (x, y, z)
    resting_half_height = pose.resting_half_height(dimensions)
    return EpisodeObject(
        family=family.identifier,
        name=f"{family.identifier}_sample",
        shape=shape,
        pose=pose.identifier,
        quaternion=_apply_yaw(pose.quaternion, yaw),
        half_size_x=x,
        half_size_y=y,
        half_size_z=z,
        mass=mass,
        friction=friction,
        safe_force=safe_force,
        x_offset=float(rng.uniform(*settings.offset_range)),
        y_offset=float(rng.uniform(*settings.offset_range)),
        yaw=yaw,
        resting_half_height=resting_half_height,
        grasp_height=pose.target_grasp_height(dimensions),
        visual_style=family.visual_style,
        evaluation_tags=family.evaluation_tags,
    )


def episode_object_from_mapping(values: Mapping[str, object]) -> EpisodeObject:
    """Build a custom deterministic object for tests, scripts, or locked suites."""

    required = (
        "shape",
        "half_size_x",
        "half_size_y",
        "half_size_z",
        "mass",
        "friction",
        "safe_force",
        "x_offset",
        "y_offset",
        "yaw",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Custom object is missing required fields: {missing}")
    shape = str(values["shape"])
    if shape not in SUPPORTED_GEOMS:
        raise ValueError(f"Unsupported object shape: {shape!r}")
    x, y, z = (float(values[key]) for key in ("half_size_x", "half_size_y", "half_size_z"))
    pose_name = str(values.get("pose", "upright"))
    base_pose = {"upright": UPRIGHT, "side_x": SIDE_X, "side_y": SIDE_Y}.get(pose_name)
    if base_pose is None:
        base_pose = StablePose(pose_name, UPRIGHT.quaternion, "z")
    yaw = float(values["yaw"])
    resting_height = float(values.get("resting_half_height", base_pose.resting_half_height((x, y, z))))
    return EpisodeObject(
        family=str(values.get("family", "custom")),
        name=str(values.get("name", "custom_object")),
        shape=shape,
        pose=pose_name,
        quaternion=_apply_yaw(base_pose.quaternion, yaw),
        half_size_x=x,
        half_size_y=y,
        half_size_z=z,
        mass=float(values["mass"]),
        friction=float(values["friction"]),
        safe_force=float(values["safe_force"]),
        x_offset=float(values["x_offset"]),
        y_offset=float(values["y_offset"]),
        yaw=yaw,
        resting_half_height=resting_height,
        grasp_height=float(
            values.get("grasp_height", max(resting_height, base_pose.minimum_grasp_height))
        ),
        visual_style=str(values.get("visual_style", "custom")),
        evaluation_tags=tuple(values.get("evaluation_tags", ("custom",))),
    )


def _sample_dimensions(
    rng: np.random.Generator, family: ObjectFamily, shape: str
) -> tuple[float, float, float]:
    x = float(rng.uniform(*family.half_size_x_range))
    z = float(rng.uniform(*family.half_size_z_range))
    if shape in {"cylinder", "capsule"}:
        y = x
    else:
        y = float(rng.uniform(*family.half_size_y_range))
    if shape == "capsule":
        z = max(z, x + 0.004)
    return x, y, z


def _intersect_range(
    base_range: FloatRange, requested_range: FloatRange | None, label: str
) -> FloatRange:
    if requested_range is None:
        return base_range
    lower = max(base_range[0], requested_range[0])
    upper = min(base_range[1], requested_range[1])
    if lower > upper:
        raise ValueError(
            f"Requested {label}_range={requested_range!r} does not overlap {base_range!r}"
        )
    return lower, upper


def _sample_safe_force(
    rng: np.random.Generator,
    family: ObjectFamily,
    settings: SamplingConfig,
    required_pad_force: float,
) -> float:
    if settings.safe_force_headroom_range is not None:
        lower, upper = settings.safe_force_headroom_range
        if lower <= 0.0 or upper < lower:
            raise ValueError("safe_force_headroom_range must be positive and ordered")
        headroom = float(rng.uniform(lower, upper))
        safe_force = required_pad_force * headroom
        return float(np.clip(safe_force, *family.safe_force_range))

    safe_min = max(family.safe_force_range[0], required_pad_force * settings.safe_force_margin)
    safe_max = max(safe_min, family.safe_force_range[1])
    return float(rng.uniform(safe_min, safe_max))


def _apply_yaw(base: Quaternion, yaw: float) -> Quaternion:
    yaw_quat = (float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0)))
    return _quaternion_multiply(yaw_quat, base)


def _quaternion_multiply(first: Quaternion, second: Quaternion) -> Quaternion:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    quaternion = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
    norm = float(np.linalg.norm(quaternion))
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


__all__ = [
    "EpisodeObject",
    "ObjectFamily",
    "SamplingConfig",
    "StablePose",
    "TOUCH_SKILL_MODES",
    "TRAINING_FAMILIES",
    "episode_object_from_mapping",
    "sample_training_object",
]
