import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from blindtouch.env import BlindTouchEnv, EnvConfig
from blindtouch.objects import SamplingConfig, TRAINING_FAMILIES


FIXED_OBJECT = {
    "shape": "cylinder",
    "half_size_x": 0.024,
    "half_size_y": 0.024,
    "half_size_z": 0.030,
    "mass": 0.10,
    "friction": 1.0,
    "safe_force": 10.0,
    "x_offset": 0.0,
    "y_offset": 0.0,
    "yaw": 0.0,
}


def test_environment_satisfies_gymnasium_contract() -> None:
    env = BlindTouchEnv()
    check_env(env, skip_render_check=True)

    observation, info = env.reset(seed=5)

    assert observation.shape == (45,)
    assert observation.dtype == np.float32
    assert env.action_space.shape == (4,)
    assert env.OBSERVATION_LAYOUT["taxels"] == slice(12, 39)
    assert info["object_params"]["family"] in TRAINING_FAMILIES
    assert "pose" in info["object_params"]
    assert info["reset_valid"]
    assert len(env.TAXEL_NAMES) == 27
    env.close()


def test_renderer_exposes_hero_and_closeup_camera_presets() -> None:
    overview = BlindTouchEnv(camera_name="overview")
    closeup = BlindTouchEnv(camera_name="closeup")
    assert overview._camera_name == "overview"
    assert closeup._camera_name == "closeup"
    overview.close()
    closeup.close()

    with pytest.raises(ValueError, match="Unsupported camera"):
        BlindTouchEnv(camera_name="missing_camera")


def test_seeded_randomization_is_reproducible_and_safe_force_is_not_observed() -> None:
    env = BlindTouchEnv()

    observation_a, info_a = env.reset(seed=13)
    observation_b, info_b = env.reset(seed=13)
    assert info_a["object_params"] == info_b["object_params"]
    np.testing.assert_allclose(observation_a, observation_b)

    fragile = {**FIXED_OBJECT, "safe_force": 0.1}
    robust = {**FIXED_OBJECT, "safe_force": 10.0}
    fragile_observation, _ = env.reset(options={"object_params": fragile})
    robust_observation, _ = env.reset(options={"object_params": robust})
    np.testing.assert_allclose(fragile_observation, robust_observation)
    env.close()


def test_randomized_friction_is_used_by_pad_object_contacts() -> None:
    env = BlindTouchEnv(config=EnvConfig(exploration_steps=0, max_episode_steps=50))
    close = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)

    for friction in (0.20, 1.10):
        env.reset(options={"object_params": {**FIXED_OBJECT, "friction": friction}})
        for _ in range(24):
            env.step(close)

        actual_friction = []
        for contact in env.data.contact[: env.data.ncon]:
            names = {env.model.geom(contact.geom1).name, env.model.geom(contact.geom2).name}
            if "object_geom" in names and any(name.endswith("_pad") for name in names):
                actual_friction.append(contact.friction[0])
        assert actual_friction
        np.testing.assert_allclose(actual_friction, friction)

    env.close()


def test_supported_shapes_reset_and_training_randomization_is_feasible() -> None:
    env = BlindTouchEnv()
    expected_types = {
        "cylinder": 5,
        "box": 6,
        "capsule": 3,
        "ellipsoid": 4,
        "chassis": 6,
    }
    for shape, geom_type in expected_types.items():
        params = {**FIXED_OBJECT, "shape": shape}
        if shape == "box" or shape == "ellipsoid":
            params["half_size_y"] = 0.027
        if shape == "capsule":
            params["half_size_x"] = params["half_size_y"] = 0.021
        if shape == "chassis":
            params["pose"] = "wheels_down"
        observation, _ = env.reset(options={"object_params": params})
        assert int(env.model.geom_type[env._object_geom_id]) == geom_type
        assert np.all(np.isfinite(observation))

    for seed in range(20):
        _, info = env.reset(seed=seed)
        params = info["object_params"]
        required_force = params["mass"] * 9.81 / (3.0 * params["friction"])
        assert params["family"] in env.sampling_config.training_families
        assert params["safe_force"] >= required_force * env.sampling_config.safe_force_margin
    env.close()


def test_chassis_enables_compound_contact_geometry_only_for_that_family() -> None:
    env = BlindTouchEnv()
    chassis_object = {
        **FIXED_OBJECT,
        "shape": "chassis",
        "pose": "wheels_down",
        "half_size_x": 0.0375,
        "half_size_y": 0.0225,
        "half_size_z": 0.0175,
    }
    _, car_info = env.reset(options={"object_params": chassis_object})
    assert car_info["object_params"]["shape"] == "chassis"
    assert all(env.model.geom_contype[geom_id] == 4 for geom_id in env._compound_geom_ids.values())
    close = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    for _ in range(24):
        _, _, terminated, truncated, car_info = env.step(close)
        if terminated or truncated:
            break
    assert car_info["peak_pad_force"] > 0.0

    env.reset(options={"object_params": FIXED_OBJECT})
    assert all(env.model.geom_contype[geom_id] == 0 for geom_id in env._compound_geom_ids.values())
    env.close()


def test_side_pose_uses_supported_resting_height_and_relative_tilt() -> None:
    side_object = {
        **FIXED_OBJECT,
        "shape": "box",
        "half_size_x": 0.027,
        "half_size_y": 0.025,
        "half_size_z": 0.018,
        "pose": "side_x",
    }
    env = BlindTouchEnv(sampling_config=SamplingConfig(training_families=("package",)))
    _, info = env.reset(options={"object_params": side_object})

    assert info["object_params"]["pose"] == "side_x"
    assert info["object_params"]["resting_half_height"] == side_object["half_size_x"]
    assert info["reset_valid"]
    assert info["tilt_radians"] < 1e-6
    env.close()


def test_exploration_phase_blocks_lifting_but_allows_finger_motion() -> None:
    env = BlindTouchEnv(config=EnvConfig(exploration_steps=3, max_episode_steps=20))
    _, initial_info = env.reset(options={"object_params": FIXED_OBJECT})
    initial_palm_target = initial_info["control_targets"][0]
    request_lift_and_close = np.ones(4, dtype=np.float32)

    for _ in range(3):
        _, _, terminated, truncated, info = env.step(request_lift_and_close)
        assert not terminated
        assert not truncated
        assert info["control_targets"][0] == initial_palm_target

    _, _, _, _, info = env.step(request_lift_and_close)
    assert info["control_targets"][0] > initial_palm_target
    assert info["control_targets"][1] > 0.0
    env.close()


def test_secure_grasp_can_lift_the_reference_object() -> None:
    env = BlindTouchEnv(config=EnvConfig(exploration_steps=0, max_episode_steps=80))
    env.reset(options={"object_params": FIXED_OBJECT})
    close = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lift = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    for _ in range(24):
        _, _, terminated, _, info = env.step(close)
        assert not terminated
    assert np.all(info["pad_forces"] > 0.0)

    for _ in range(40):
        _, _, terminated, truncated, info = env.step(lift)
        if terminated or truncated:
            break

    assert terminated
    assert not truncated
    assert info["outcome"] == "success"
    assert info["lift_height"] >= env.config.lift_target_height
    env.close()


def test_reward_shaping_prefers_balanced_touch_to_inaction() -> None:
    env = BlindTouchEnv()
    env.reset(options={"object_params": FIXED_OBJECT})
    env._step_count = 12
    action = np.zeros(4, dtype=np.float32)
    no_touch = np.zeros(3, dtype=np.float32)
    balanced_touch = np.array([0.12, 0.11, 0.10], dtype=np.float32)

    no_touch_reward = env._reward(
        action=action,
        pad_forces=no_touch,
        lift_height=0.0,
        slipped=False,
        damaged=False,
        dropped=False,
        unstable=False,
        succeeded=False,
        lift_allowed=False,
        grip_metrics=env._grip_metrics(no_touch),
    )
    touch_metrics = env._grip_metrics(balanced_touch)
    touch_reward = env._reward(
        action=action,
        pad_forces=balanced_touch,
        lift_height=0.0,
        slipped=False,
        damaged=False,
        dropped=False,
        unstable=False,
        succeeded=False,
        lift_allowed=False,
        grip_metrics=touch_metrics,
    )

    assert touch_metrics["contact_count"] == 3
    assert touch_metrics["grip_score"] > 0.0
    assert touch_reward > no_touch_reward
    env.close()


def test_reward_shaping_discourages_stalled_grip_after_exploration() -> None:
    env = BlindTouchEnv()
    env.reset(options={"object_params": FIXED_OBJECT})
    env._step_count = env.config.exploration_steps + 20
    env._previous_lift_height = 0.0
    balanced_touch = np.array([0.12, 0.11, 0.10], dtype=np.float32)
    metrics = env._grip_metrics(balanced_touch)

    stalled_reward = env._reward(
        action=np.zeros(4, dtype=np.float32),
        pad_forces=balanced_touch,
        lift_height=0.0,
        slipped=False,
        damaged=False,
        dropped=False,
        unstable=False,
        succeeded=False,
        lift_allowed=True,
        grip_metrics=metrics,
    )
    lifting_reward = env._reward(
        action=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        pad_forces=balanced_touch,
        lift_height=0.002,
        slipped=False,
        damaged=False,
        dropped=False,
        unstable=False,
        succeeded=False,
        lift_allowed=True,
        grip_metrics=metrics,
    )

    assert metrics["grip_score"] >= 0.45
    assert lifting_reward > stalled_reward
    env.close()


def test_timeout_penalty_marks_no_lift_timeouts_as_expensive() -> None:
    env = BlindTouchEnv()
    env.reset(options={"object_params": FIXED_OBJECT})
    env._max_contact_count = 3
    env._max_lift_height = 0.0
    no_lift_penalty = env._timeout_penalty()

    env._max_lift_height = env.config.lift_target_height * 0.80
    partial_lift_penalty = env._timeout_penalty()

    assert no_lift_penalty == pytest.approx(
        env.config.timeout_penalty + env.config.no_lift_timeout_penalty
    )
    assert partial_lift_penalty < no_lift_penalty
    env.close()


def test_force_limit_terminates_a_fragile_grasp_as_damage() -> None:
    fragile_object = {**FIXED_OBJECT, "safe_force": 0.1}
    env = BlindTouchEnv(config=EnvConfig(exploration_steps=0, max_episode_steps=50))
    env.reset(options={"object_params": fragile_object})
    close = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)

    for _ in range(30):
        _, reward, terminated, truncated, info = env.step(close)
        if terminated or truncated:
            break

    assert terminated
    assert not truncated
    assert info["outcome"] == "damage"
    assert info["peak_pad_force"] > fragile_object["safe_force"]
    assert reward < -5.0
    env.close()


def test_random_actions_remain_numerically_stable() -> None:
    env = BlindTouchEnv()
    observation, _ = env.reset(seed=21)

    for _ in range(250):
        observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(observation))
        assert np.isfinite(reward)
        if terminated or truncated:
            observation, _ = env.reset()

    env.close()
