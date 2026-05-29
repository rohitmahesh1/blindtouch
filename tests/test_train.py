import numpy as np
import pytest

from blindtouch.env import EnvConfig
from blindtouch.evaluate import EvaluationCase, EvaluationSuite
from blindtouch.objects import episode_object_from_mapping
import blindtouch.train as train_module
from blindtouch.train import (
    FEASIBILITY_GATE_MESSAGE,
    StackedPolicyController,
    TrainingConfig,
    algorithm_hyperparameters,
    curriculum_sampling_config,
    evaluate_policy,
    make_training_env,
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
    assert ppo["learning_rate"] == pytest.approx(3e-4)
    assert sac["learning_rate"] == pytest.approx(1e-4)
    assert ppo["n_steps"] == 2048
    assert ppo["gae_lambda"] == pytest.approx(0.95)
    assert sac["buffer_size"] == 1_000_000
    assert sac["learning_starts"] == 10_000
    assert sac["ent_coef"] == "auto"
    env.close()


def test_curriculum_expands_poses_then_adds_compound_chassis_family() -> None:
    robust = make_training_env(curriculum_stage="robust_upright")
    robust_metadata = [robust.reset(seed=seed)[1]["object_params"] for seed in range(30)]
    assert all(metadata["pose"] == "upright" for metadata in robust_metadata)
    assert all(metadata["family"] in {"rounded", "container"} for metadata in robust_metadata)
    assert all(metadata["friction"] >= 0.65 for metadata in robust_metadata)
    assert all(metadata["mass"] <= 0.110 for metadata in robust_metadata)
    robust.close()

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
    robust_stage = curriculum_sampling_config("robust_upright")
    assert robust_stage.training_families == ("rounded", "container")
    assert robust_stage.friction_range == (0.65, 1.20)
    assert robust_stage.mass_range == (0.030, 0.110)
    assert robust_stage.safe_force_margin == pytest.approx(4.0)
    fragile_stage = curriculum_sampling_config("fragile_upright")
    assert fragile_stage.training_families == ("rounded", "container", "package")
    assert fragile_stage.allowed_poses == ("upright",)
    assert fragile_stage.safe_force_headroom_range == (2.60, 3.80)


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
    require_feasibility_acknowledgement("robust_upright", False)
    with pytest.raises(RuntimeError, match="upright stage-1"):
        require_feasibility_acknowledgement("fragile_upright", False)
    with pytest.raises(RuntimeError, match="upright stage-1"):
        require_feasibility_acknowledgement("all_poses", False)
    with pytest.raises(RuntimeError, match="upright stage-1"):
        require_feasibility_acknowledgement("with_chassis", False)
    require_feasibility_acknowledgement("with_chassis", True)
    assert "currently certified" in FEASIBILITY_GATE_MESSAGE
    assert TrainingConfig("ppo").allow_uncertified_environment is False


def test_sac_bc_anchor_requires_warm_start_demonstrations() -> None:
    with pytest.raises(ValueError, match="requires warm_start_transitions"):
        TrainingConfig("sac", sac_bc_anchor_weight=1.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        TrainingConfig("sac", warm_start_transitions=1, sac_bc_anchor_weight=-1.0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainingConfig(
            "sac",
            warm_start_transitions=1,
            sac_bc_anchor_weight=1.0,
            sac_bc_anchor_batch_size=0,
        )


def test_zero_step_training_is_reserved_for_warm_start_gates() -> None:
    config = TrainingConfig("ppo", total_timesteps=0, warm_start_transitions=1)
    assert config.total_timesteps == 0
    with pytest.raises(ValueError, match="total_timesteps=0 requires"):
        TrainingConfig("ppo", total_timesteps=0)
    with pytest.raises(ValueError, match="total_timesteps cannot be negative"):
        TrainingConfig("ppo", total_timesteps=-1)


def test_warm_start_teacher_can_use_composed_touch_prior() -> None:
    assert TrainingConfig("ppo").warm_start_teacher == "safe_force"
    assert TrainingConfig("ppo", warm_start_teacher="composed_touch").warm_start_teacher == (
        "composed_touch"
    )
    assert train_module._warm_start_curriculum_stages(TrainingConfig("ppo")) == ("upright",)
    assert train_module._warm_start_curriculum_stages(
        TrainingConfig("ppo", warm_start_profile="fragile_mix")
    ) == ("upright", "fragile_upright")
    assert train_module._warm_start_curriculum_stages(
        TrainingConfig(
            "ppo",
            curriculum_stage="fragile_upright",
            warm_start_profile="fragile_mix",
        )
    ) == ("fragile_upright",)
    with pytest.raises(ValueError, match="Unsupported warm-start teacher"):
        TrainingConfig("ppo", warm_start_teacher="oracle")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported warm-start profile"):
        TrainingConfig("ppo", warm_start_profile="oracle")  # type: ignore[arg-type]


def test_sac_bc_anchor_configuration_attaches_demonstrations() -> None:
    class DummyAnchoredSac:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def set_bc_anchor(self, observations, actions, **kwargs) -> None:
            self.calls.append(
                {
                    "observations": observations,
                    "actions": actions,
                    **kwargs,
                }
            )

    model = DummyAnchoredSac()
    observations = np.zeros((3, 360), dtype=np.float32)
    actions = np.zeros((3, 4), dtype=np.float32)
    config = TrainingConfig(
        "sac",
        seed=7,
        warm_start_transitions=3,
        sac_bc_anchor_weight=2.5,
        sac_bc_anchor_batch_size=64,
    )

    train_module._configure_sac_bc_anchor(model, config, observations, actions)

    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["observations"] is observations
    assert call["actions"] is actions
    assert call["weight"] == pytest.approx(2.5)
    assert call["batch_size"] == 64
    assert call["seed"] == 41007


def test_history_safe_force_teacher_uses_only_policy_observation_history() -> None:
    frames = np.zeros((8, 45), dtype=np.float32)
    close_action = train_module._history_safe_force_teacher_action(frames.reshape(-1))
    np.testing.assert_array_equal(
        close_action, np.array([0.0, 0.18, 0.18, 0.18], dtype=np.float32)
    )

    frames[:, 12] = 0.34 / 5.0
    frames[:, 21] = 0.34 / 5.0
    frames[:, 43] = 1.0
    lift_action = train_module._history_safe_force_teacher_action(frames.reshape(-1))
    np.testing.assert_array_equal(
        lift_action, np.array([1.0, 0.0, 0.0, 0.08], dtype=np.float32)
    )

    frames[-1, 12] = 0.60 / 5.0
    release_action = train_module._history_safe_force_teacher_action(frames.reshape(-1))
    assert release_action[0] == 1.0
    assert release_action[1] < 0.0


def test_composed_touch_teacher_starts_from_policy_observation_history() -> None:
    teacher = train_module._ComposedTouchTeacher()
    stacked = np.zeros((8, 45), dtype=np.float32).reshape(-1)
    first_action = teacher.act(stacked)
    np.testing.assert_array_equal(
        first_action, np.array([0.0, 0.26, 0.26, 0.26], dtype=np.float32)
    )

    contacted = np.zeros((8, 45), dtype=np.float32)
    contacted[-1, 12] = 0.50 / 5.0
    contacted[-1, 21] = 0.50 / 5.0
    contacted[-1, 30] = 0.50 / 5.0
    for _ in range(3):
        action = teacher.act(contacted.reshape(-1))
    assert action[0] >= 0.0
    assert teacher.selected_branch in {
        "round_retention",
        "rigid_asymmetric",
        "slippery_retention",
        "fragile_balance",
    }


def test_learned_policy_evaluation_uses_360_values_and_writes_reports(tmp_path) -> None:
    policy = RecordingPolicy()
    episode_object = episode_object_from_mapping(
        {
            "shape": "cylinder",
            "half_size_x": 0.024,
            "half_size_y": 0.024,
            "half_size_z": 0.030,
            "mass": 0.10,
            "friction": 1.0,
            "safe_force": 1.0,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "yaw": 0.0,
            "name": "reference_object",
        }
    )
    suite = EvaluationSuite("reference", (EvaluationCase("reference", 50_000, episode_object),))
    records = evaluate_policy(
        policy,
        suite,
        algorithm="ppo",
        checkpoint="dry.zip",
        output_prefix=tmp_path / "ppo_reference",
        env_config=EnvConfig(exploration_steps=0, max_episode_steps=2),
    )

    assert records[0]["controller"] == "ppo"
    assert records[0]["checkpoint"] == "dry.zip"
    assert records[0]["outcome"] == "timeout"
    assert policy.observations
    assert all(observation.shape == (360,) for observation in policy.observations)
    assert (tmp_path / "ppo_reference.csv").exists()
    assert (tmp_path / "ppo_reference.jsonl").exists()
