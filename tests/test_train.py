import numpy as np
import pytest

from blindtouch.env import EnvConfig
from blindtouch.evaluate import EvaluationSuite, ReplayResult, demo_case
import blindtouch.train as train_module
from blindtouch.train import (
    FEASIBILITY_GATE_MESSAGE,
    StackedPolicyController,
    TrainingConfig,
    algorithm_hyperparameters,
    curriculum_sampling_config,
    evaluate_policy,
    make_training_env,
    render_learned_policy_replay,
    require_feasibility_acknowledgement,
)


class RecordingPolicy:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        assert deterministic
        self.observations.append(observation.copy())
        return np.zeros(4, dtype=np.float32), None


def test_training_contract_uses_stacked_touch_observations_for_both_algorithms() -> None:
    env = make_training_env()
    observation, _ = env.reset(seed=3)
    assert observation.shape == (360,)

    ppo = algorithm_hyperparameters("ppo")
    sac = algorithm_hyperparameters("sac")
    assert ppo["policy_kwargs"] == sac["policy_kwargs"] == {"net_arch": [256, 256]}
    assert ppo["n_steps"] == 2048
    assert ppo["gae_lambda"] == pytest.approx(0.95)
    assert sac["buffer_size"] == 1_000_000
    assert sac["learning_starts"] == 10_000
    assert sac["ent_coef"] == "auto"
    env.close()


def test_curriculum_expands_poses_then_adds_compound_chassis_family() -> None:
    upright = make_training_env(curriculum_stage="upright")
    upright_metadata = [upright.reset(seed=seed)[1]["object_params"] for seed in range(30)]
    assert all(metadata["pose"] == "upright" for metadata in upright_metadata)
    assert all(metadata["family"] != "chassis" for metadata in upright_metadata)
    upright.close()

    all_poses = make_training_env(curriculum_stage="all_poses")
    pose_metadata = [all_poses.reset(seed=seed)[1]["object_params"] for seed in range(60)]
    assert any(metadata["pose"] in {"side_x", "side_y"} for metadata in pose_metadata)
    assert all(metadata["family"] != "chassis" for metadata in pose_metadata)
    all_poses.close()

    with_chassis = make_training_env(curriculum_stage="with_chassis")
    family_metadata = [with_chassis.reset(seed=seed)[1]["object_params"] for seed in range(60)]
    assert any(metadata["family"] == "chassis" for metadata in family_metadata)
    with_chassis.close()
    stage_one = curriculum_sampling_config("upright")
    assert stage_one.allowed_poses == ("upright",)
    assert stage_one.offset_range == (-0.002, 0.002)
    assert stage_one.safe_force_margin == pytest.approx(3.0)


def test_stacked_policy_controller_matches_training_history_layout() -> None:
    policy = RecordingPolicy()
    controller = StackedPolicyController(policy)
    initial = np.arange(45, dtype=np.float32) / 100.0
    later = np.full(45, 0.5, dtype=np.float32)

    controller.reset(initial, {})
    np.testing.assert_array_equal(controller.act(initial, {}), np.zeros(4, dtype=np.float32))
    controller.act(later, {})

    assert policy.observations[0].shape == (360,)
    initial_frames = policy.observations[0].reshape(8, 45)
    np.testing.assert_allclose(initial_frames, np.repeat(initial[None, :], 8, axis=0))
    later_frames = policy.observations[1].reshape(8, 45)
    np.testing.assert_allclose(later_frames[:-1], initial_frames[1:])
    np.testing.assert_allclose(later_frames[-1], later)


def test_learning_is_restricted_to_certified_stage_without_acknowledgement() -> None:
    require_feasibility_acknowledgement("upright", False)
    with pytest.raises(RuntimeError, match="upright stage-1"):
        require_feasibility_acknowledgement("all_poses", False)
    with pytest.raises(RuntimeError, match="upright stage-1"):
        require_feasibility_acknowledgement("with_chassis", False)
    require_feasibility_acknowledgement("with_chassis", True)
    assert "currently certified" in FEASIBILITY_GATE_MESSAGE
    assert TrainingConfig("ppo").allow_uncertified_environment is False


def test_learned_policy_evaluation_uses_360_values_and_writes_reports(tmp_path) -> None:
    policy = RecordingPolicy()
    suite = EvaluationSuite("demo", (demo_case("orange"),))
    records = evaluate_policy(
        policy,
        suite,
        algorithm="ppo",
        checkpoint="dry.zip",
        output_prefix=tmp_path / "ppo_demo",
        env_config=EnvConfig(exploration_steps=0, max_episode_steps=2),
    )

    assert records[0]["controller"] == "ppo"
    assert records[0]["checkpoint"] == "dry.zip"
    assert records[0]["outcome"] == "timeout"
    assert policy.observations
    assert all(observation.shape == (360,) for observation in policy.observations)
    assert (tmp_path / "ppo_demo.csv").exists()
    assert (tmp_path / "ppo_demo.jsonl").exists()


def test_learned_policy_replay_uses_loaded_policy_history_and_writes_frames(
    tmp_path, monkeypatch
) -> None:
    policy = RecordingPolicy()

    def fake_render_replay(controller_factory, case, **kwargs):
        controller = controller_factory()
        observation = np.arange(45, dtype=np.float32) / 100.0
        controller.reset(observation, {})
        action = controller.act(observation, {})
        np.testing.assert_array_equal(action, np.zeros(4, dtype=np.float32))
        assert case.object.name == "orange"
        assert kwargs["controller_name"] == "ppo"
        assert kwargs["checkpoint"] == "dry.zip"
        assert kwargs["output_dir"] == tmp_path
        return ReplayResult(
            {"controller": "ppo", "checkpoint": "dry.zip", "object_name": "orange"},
            3,
            tmp_path / "orange_ppo_overview_frames",
            None,
        )

    monkeypatch.setattr(train_module, "render_replay", fake_render_replay)
    result = render_learned_policy_replay(
        policy,
        algorithm="ppo",
        checkpoint="dry.zip",
        object_name="orange",
        output_dir=tmp_path,
        env_config=EnvConfig(exploration_steps=0, max_episode_steps=2),
        width=160,
        height=120,
        encode_video=False,
    )

    assert result.record["controller"] == "ppo"
    assert result.record["checkpoint"] == "dry.zip"
    assert result.record["object_name"] == "orange"
    assert result.frame_count > 0
    assert result.video_path is None
    assert result.frame_directory == tmp_path / "orange_ppo_overview_frames"
    assert policy.observations
    assert all(observation.shape == (360,) for observation in policy.observations)
