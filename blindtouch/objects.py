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

    training_families: tuple[str, ...] = ("rounded", "container", "package", "chassis")
    allowed_poses: tuple[str, ...] | None = None
    offset_range: FloatRange = (-0.006, 0.006)
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
ORIENTATION_FREE = StablePose("orientation_free", (1.0, 0.0, 0.0, 0.0), "z")
WHEELS_DOWN = StablePose("wheels_down", (1.0, 0.0, 0.0, 0.0), "z", (-0.35, 0.35))
EDGE_RESTING = StablePose("edge_resting", SIDE_Y.quaternion, "y", (-0.7, 0.7))
CAR_SIDE_RESTING = StablePose("side_resting", SIDE_Y.quaternion, "y", (-0.25, 0.25))


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
        evaluation_tags=("training", "rounded"),
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
        evaluation_tags=("training", "container"),
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
        evaluation_tags=("training", "package"),
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
        evaluation_tags=("training", "compound"),
    )
}


DEMO_OBJECT_POSES: dict[str, dict[str, EpisodeObject]] = {
    "orange": {
        "orientation_free": EpisodeObject(
        family="household",
        name="orange",
        shape="ellipsoid",
        pose="orientation_free",
        quaternion=ORIENTATION_FREE.quaternion,
        half_size_x=0.031,
        half_size_y=0.031,
        half_size_z=0.031,
        mass=0.120,
        friction=0.55,
        safe_force=0.80,
        x_offset=0.0,
        y_offset=0.0,
        yaw=0.0,
        resting_half_height=0.031,
        grasp_height=0.031,
        visual_style="orange",
        evaluation_tags=("demo", "held_out", "gentle"),
        )
    },
    "soap_bar": {
        "broad_face": EpisodeObject(
        family="household",
        name="soap_bar",
        shape="box",
        pose="broad_face",
        quaternion=UPRIGHT.quaternion,
        half_size_x=0.0325,
        half_size_y=0.021,
        half_size_z=0.0125,
        mass=0.095,
        friction=0.22,
        safe_force=2.00,
        x_offset=0.0,
        y_offset=0.0,
        yaw=0.0,
        resting_half_height=0.0125,
        grasp_height=0.020,
        visual_style="soap_bar",
        evaluation_tags=("demo", "held_out", "slippery"),
        ),
        "edge_resting": EpisodeObject(
            family="household",
            name="soap_bar",
            shape="box",
            pose="edge_resting",
            quaternion=EDGE_RESTING.quaternion,
            half_size_x=0.0325,
            half_size_y=0.021,
            half_size_z=0.0125,
            mass=0.095,
            friction=0.22,
            safe_force=2.00,
            x_offset=0.0,
            y_offset=0.0,
            yaw=0.0,
            resting_half_height=0.021,
            grasp_height=0.021,
            visual_style="soap_bar",
            evaluation_tags=("demo", "held_out", "slippery", "alternate_pose"),
        ),
    },
    "tomato": {
        "orientation_free": EpisodeObject(
        family="household",
        name="tomato",
        shape="ellipsoid",
        pose="orientation_free",
        quaternion=ORIENTATION_FREE.quaternion,
        half_size_x=0.030,
        half_size_y=0.030,
        half_size_z=0.028,
        mass=0.100,
        friction=0.80,
        safe_force=0.55,
        x_offset=0.0,
        y_offset=0.0,
        yaw=0.0,
        resting_half_height=0.028,
        grasp_height=0.028,
        visual_style="tomato",
        evaluation_tags=("demo", "held_out", "fragile"),
        )
    },
    "toy_car": {
        "wheels_down": EpisodeObject(
        family="household",
        name="toy_car",
        shape="chassis",
        pose="wheels_down",
        quaternion=WHEELS_DOWN.quaternion,
        half_size_x=0.0375,
        half_size_y=0.0225,
        half_size_z=0.0175,
        mass=0.090,
        friction=0.55,
        safe_force=1.80,
        x_offset=0.0,
        y_offset=0.0,
        yaw=0.0,
        resting_half_height=0.0175,
        grasp_height=0.022,
        visual_style="toy_car",
        evaluation_tags=("demo", "held_out", "compound_collision"),
        ),
        "side_resting": EpisodeObject(
            family="household",
            name="toy_car",
            shape="chassis",
            pose="side_resting",
            quaternion=CAR_SIDE_RESTING.quaternion,
            half_size_x=0.0375,
            half_size_y=0.0225,
            half_size_z=0.0175,
            mass=0.090,
            friction=0.55,
            safe_force=1.80,
            x_offset=0.0,
            y_offset=0.0,
            yaw=0.0,
            resting_half_height=0.0225,
            grasp_height=0.0225,
            visual_style="toy_car",
            evaluation_tags=("demo", "held_out", "compound_collision", "alternate_pose"),
        ),
    },
}

DEMO_DEFAULT_POSES = {
    "orange": "orientation_free",
    "soap_bar": "broad_face",
    "tomato": "orientation_free",
    "toy_car": "wheels_down",
}
DEMO_PROXIES = {
    name: DEMO_OBJECT_POSES[name][pose] for name, pose in DEMO_DEFAULT_POSES.items()
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
    friction = float(rng.uniform(*family.friction_range))
    feasible_mass_ceiling = (
        3.0
        * friction
        * settings.nominal_pad_force_capacity
        * settings.holding_force_margin
        / 9.81
    )
    upper_mass = max(family.mass_range[0], min(family.mass_range[1], feasible_mass_ceiling))
    mass = float(rng.uniform(family.mass_range[0], upper_mass))
    required_pad_force = mass * 9.81 / (3.0 * friction)
    safe_min = max(family.safe_force_range[0], required_pad_force * settings.safe_force_margin)
    safe_max = max(safe_min, family.safe_force_range[1])
    safe_force = float(rng.uniform(safe_min, safe_max))
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


def get_demo_object(name: str, pose: str | None = None) -> EpisodeObject:
    """Return a held-out household object in a deterministic stable pose."""

    if name not in DEMO_OBJECT_POSES:
        raise ValueError(f"Unknown demo object: {name!r}")
    selected_pose = pose or DEMO_DEFAULT_POSES[name]
    if selected_pose not in DEMO_OBJECT_POSES[name]:
        valid = tuple(DEMO_OBJECT_POSES[name])
        raise ValueError(f"Object {name!r} does not support pose {selected_pose!r}; valid poses: {valid}")
    return DEMO_OBJECT_POSES[name][selected_pose]


def demo_object_poses(name: str) -> tuple[str, ...]:
    """List the deterministic stable poses available for a named demo object."""

    if name not in DEMO_OBJECT_POSES:
        raise ValueError(f"Unknown demo object: {name!r}")
    return tuple(DEMO_OBJECT_POSES[name])


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
    "DEMO_PROXIES",
    "DEMO_DEFAULT_POSES",
    "DEMO_OBJECT_POSES",
    "EpisodeObject",
    "ObjectFamily",
    "SamplingConfig",
    "StablePose",
    "TRAINING_FAMILIES",
    "demo_object_poses",
    "episode_object_from_mapping",
    "get_demo_object",
    "sample_training_object",
]
