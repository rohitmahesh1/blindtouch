import numpy as np

from blindtouch.env import BlindTouchEnv, EnvConfig
from blindtouch.objects import DEMO_OBJECT_POSES, demo_object_poses


SAFE_LIFT_SEQUENCES = {
    "orange": ((0.25, 66),),
    "soap_bar": ((0.20, 84),),
    "tomato": ((0.40, 38), (0.05, 23)),
    "toy_car": ((0.20, 80),),
}


def test_demo_catalog_exposes_video_objects_and_alternate_poses() -> None:
    assert set(DEMO_OBJECT_POSES) == {"orange", "toy_car", "soap_bar", "tomato"}
    assert demo_object_poses("soap_bar") == ("broad_face", "edge_resting")
    assert demo_object_poses("toy_car") == ("wheels_down", "side_resting")


def test_demo_styles_activate_recognizable_materials_and_decorations() -> None:
    expected_materials = {
        "orange": "orange_skin",
        "tomato": "tomato_skin",
        "toy_car": "car_body",
    }
    env = BlindTouchEnv()
    for name, material in expected_materials.items():
        env.reset(options={"demo_object": name})
        material_id = env._material_ids[material]
        assert env.model.geom_matid[env._object_geom_id] == material_id
        np.testing.assert_allclose(
            env.model.geom_rgba[env._object_geom_id],
            env.model.mat_rgba[material_id],
        )
        assert env.model.geom_matid[env._accent_geom_ids[0]] >= 0

    env.reset(options={"demo_object": "soap_bar"})
    assert env.model.geom_rgba[env._object_geom_id, 3] == 0.0
    assert env.model.geom_matid[env._accent_geom_ids[0]] == env._material_ids["soap_body"]
    assert env.model.geom_matid[env._accent_geom_ids[1]] == env._material_ids["soap_stamp"]
    np.testing.assert_allclose(
        env.model.geom_rgba[env._accent_geom_ids[0]],
        env.model.mat_rgba[env._material_ids["soap_body"]],
    )

    env.reset(options={"demo_object": "toy_car"})
    assert len(env._accent_geom_ids) >= 10
    assert env.model.geom_matid[env._accent_geom_ids[0]] == env._material_ids["car_lamp"]
    assert env.model.geom_matid[env._accent_geom_ids[2]] == env._material_ids["car_tail_lamp"]
    assert (
        env.model.geom_matid[env._accent_geom_ids[4]]
        == env._material_ids["car_racing_stripe"]
    )
    assert (
        env.model.geom_matid[env._accent_geom_ids[5]]
        == env._material_ids["car_racing_stripe"]
    )
    assert env.model.geom_matid[env._accent_geom_ids[6]] == env._material_ids["car_hubcap"]
    env.close()


def test_toy_car_runtime_geometry_is_not_collapsed_to_body_origin() -> None:
    env = BlindTouchEnv()
    env.reset(options={"demo_object": "toy_car"})

    body_world = env.data.geom_xpos[env._object_geom_id]
    cabin_world = env.data.geom_xpos[env._compound_geom_ids["object_cabin_geom"]]
    wheel_world = env.data.geom_xpos[env._compound_geom_ids["object_wheel_fl_geom"]]
    lamp_world = env.data.geom_xpos[env._accent_geom_ids[0]]

    assert cabin_world[2] > body_world[2] + 0.005
    assert abs(wheel_world[0] - body_world[0]) > 0.02
    assert abs(wheel_world[1] - body_world[1]) > 0.01
    assert abs(lamp_world[0] - body_world[0]) > 0.02
    env.close()


def test_gentle_probe_produces_taxel_readings_for_every_demo_pose() -> None:
    env = BlindTouchEnv(config=EnvConfig(exploration_steps=0, max_episode_steps=260))
    probe = np.array([0.0, 0.10, 0.10, 0.10], dtype=np.float32)
    taxel_slice = env.OBSERVATION_LAYOUT["taxels"]

    for name, poses in DEMO_OBJECT_POSES.items():
        for pose in poses:
            observation, _ = env.reset(options={"demo_object": name, "pose": pose})
            peak_taxel = 0.0
            for _ in range(210):
                observation, _, terminated, truncated, _ = env.step(probe)
                peak_taxel = max(peak_taxel, float(np.max(observation[taxel_slice])))
                if peak_taxel > 0.0 or terminated or truncated:
                    break
            assert peak_taxel > 0.0, f"No tactile reading for {name}/{pose}"
    env.close()


def test_hand_authored_sequences_safely_lift_each_default_demo_object() -> None:
    env = BlindTouchEnv(config=EnvConfig(exploration_steps=0, max_episode_steps=120))
    lift = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    for name, close_phases in SAFE_LIFT_SEQUENCES.items():
        _, info = env.reset(options={"demo_object": name})
        terminated = truncated = False
        for close_amplitude, close_steps in close_phases:
            close = np.array(
                [0.0, close_amplitude, close_amplitude, close_amplitude], dtype=np.float32
            )
            for _ in range(close_steps):
                _, _, terminated, truncated, info = env.step(close)
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
        if not (terminated or truncated):
            for _ in range(25):
                _, _, terminated, truncated, info = env.step(lift)
                if terminated or truncated:
                    break

        assert info["outcome"] == "success", f"Unsafe feasibility sequence for {name}: {info}"
        assert info["peak_pad_force"] <= info["object_params"]["safe_force"]
    env.close()
