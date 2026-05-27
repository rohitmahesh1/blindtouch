import numpy as np
import pytest

from blindtouch.objects import (
    TRAINING_FAMILIES,
    SamplingConfig,
    episode_object_from_mapping,
    get_demo_object,
    sample_training_object,
)


def test_training_catalog_declares_auditable_families_and_stable_poses() -> None:
    assert set(TRAINING_FAMILIES) == {"rounded", "container", "package", "chassis"}
    assert TRAINING_FAMILIES["rounded"].collision_kinds == ("ellipsoid",)
    assert TRAINING_FAMILIES["container"].collision_kinds == ("cylinder", "capsule")
    assert TRAINING_FAMILIES["package"].collision_kinds == ("box",)
    assert {pose.identifier for pose in TRAINING_FAMILIES["container"].stable_poses} == {
        "upright",
        "side_x",
    }
    assert TRAINING_FAMILIES["package"].stable_poses[0].target_grasp_height(
        (0.025, 0.025, 0.010)
    ) == pytest.approx(0.020)
    assert TRAINING_FAMILIES["chassis"].collision_kinds == ("chassis",)
    assert TRAINING_FAMILIES["chassis"].stable_poses[0].identifier == "wheels_down"


def test_training_sampling_is_seeded_physical_and_includes_side_poses() -> None:
    first = sample_training_object(np.random.default_rng(31))
    second = sample_training_object(np.random.default_rng(31))
    assert first == second

    poses = set()
    for _ in range(500):
        sample = sample_training_object(np.random.default_rng(_))
        poses.add(sample.pose)
        assert sample.family in TRAINING_FAMILIES
        assert np.isclose(np.linalg.norm(sample.quaternion), 1.0)
        required_pad_force = sample.mass * 9.81 / (3.0 * sample.friction)
        assert sample.safe_force >= required_pad_force * SamplingConfig().safe_force_margin
    assert {"side_x", "side_y"} <= poses


def test_demo_objects_lock_household_specs_and_named_poses() -> None:
    orange = get_demo_object("orange")
    soap_edge = get_demo_object("soap_bar", "edge_resting")
    tomato = get_demo_object("tomato")
    car = get_demo_object("toy_car", "wheels_down")

    assert orange.shape == "ellipsoid"
    assert orange.mass == pytest.approx(0.120)
    assert orange.friction == pytest.approx(0.55)
    assert orange.safe_force == pytest.approx(0.80)
    assert soap_edge.shape == "box"
    assert soap_edge.pose == "edge_resting"
    assert tomato.shape == "ellipsoid"
    assert tomato.friction == pytest.approx(0.80)
    assert tomato.safe_force == pytest.approx(0.55)
    assert car.shape == "chassis"
    assert "compound_collision" in car.evaluation_tags
    with pytest.raises(ValueError):
        get_demo_object("toy_car", "side_x")


def test_custom_pose_computes_resting_axis_and_orientation() -> None:
    custom = episode_object_from_mapping(
        {
            "shape": "box",
            "half_size_x": 0.031,
            "half_size_y": 0.024,
            "half_size_z": 0.017,
            "mass": 0.10,
            "friction": 0.5,
            "safe_force": 1.0,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "yaw": 0.25,
            "pose": "side_y",
        }
    )

    assert custom.resting_half_height == pytest.approx(custom.half_size_y)
    assert custom.grasp_height == pytest.approx(custom.half_size_y)
    assert np.isclose(np.linalg.norm(custom.quaternion), 1.0)
