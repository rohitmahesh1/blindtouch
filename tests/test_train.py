import json
import random

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
    conservative = algorithm_hyperparameters(
        "ppo",
        TrainingConfig(
            "ppo",
            learning_rate=5e-5,
            ppo_clip_range=0.05,
            ppo_target_kl=0.02,
        ),
    )
    assert conservative["learning_rate"] == pytest.approx(5e-5)
    assert conservative["clip_range"] == pytest.approx(0.05)
    assert conservative["target_kl"] == pytest.approx(0.02)
    delayed_sac = algorithm_hyperparameters(
        "sac",
        TrainingConfig(
            "sac",
            learning_rate=5e-5,
            sac_learning_starts=20_000,
            sac_ent_coef=0.01,
        ),
    )
    assert delayed_sac["learning_rate"] == pytest.approx(5e-5)
    assert delayed_sac["learning_starts"] == 20_000
    assert delayed_sac["ent_coef"] == pytest.approx(0.01)
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
    assert stage_one.training_families == (
        "rounded",
        "container",
        "package",
        "slippery",
        "fragile",
    )
    assert stage_one.allowed_poses == ("upright",)
    assert stage_one.offset_range == (-0.002, 0.002)
    assert stage_one.safe_force_margin == pytest.approx(3.0)
    robust_stage = curriculum_sampling_config("robust_upright")
    assert robust_stage.training_families == ("rounded", "container")
    assert robust_stage.friction_range == (0.65, 1.20)
    assert robust_stage.mass_range == (0.030, 0.110)
    assert robust_stage.safe_force_margin == pytest.approx(4.0)
    fragile_stage = curriculum_sampling_config("fragile_upright")
    assert fragile_stage.training_families == ("fragile", "rounded", "container", "package")
    assert fragile_stage.allowed_poses == ("upright",)
    assert fragile_stage.safe_force_headroom_range == (2.60, 3.80)
    assert TrainingConfig("ppo").evaluation_suites == (
        "validation_procedural",
        "test_procedural_holdout",
    )
    assert TrainingConfig("ppo").promotion_suite == "test_procedural_holdout"
    with pytest.raises(ValueError, match="evaluation_suites"):
        TrainingConfig("ppo", evaluation_suites=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicates"):
        TrainingConfig(
            "ppo",
            evaluation_suites=("validation_procedural", "validation_procedural"),
        )
    with pytest.raises(ValueError, match="Unsupported evaluation"):
        TrainingConfig("ppo", evaluation_suites=("demo",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="promotion_suite"):
        TrainingConfig(
            "ppo",
            evaluation_suites=("validation_procedural",),
            promotion_suite="test_procedural_holdout",
        )


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
    with pytest.raises(ValueError, match="requires warm_start_transitions"):
        TrainingConfig("sac", sac_policy_anchor_weight=1.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        TrainingConfig(
            "sac",
            warm_start_transitions=1,
            sac_policy_anchor_weight=-1.0,
        )


def test_zero_step_training_is_reserved_for_warm_start_gates() -> None:
    config = TrainingConfig("ppo", total_timesteps=0, warm_start_transitions=1)
    assert config.total_timesteps == 0
    with pytest.raises(ValueError, match="total_timesteps=0 requires"):
        TrainingConfig("ppo", total_timesteps=0)
    with pytest.raises(ValueError, match="total_timesteps cannot be negative"):
        TrainingConfig("ppo", total_timesteps=-1)


def test_warm_start_teacher_can_use_composed_touch_prior() -> None:
    assert TrainingConfig("ppo").warm_start_teacher == "composed_touch"
    assert TrainingConfig("ppo").warm_start_learning_rate is None
    assert train_module._warm_start_learning_rate(
        TrainingConfig("ppo", learning_rate=2e-5)
    ) == pytest.approx(3e-4)
    assert train_module._warm_start_learning_rate(
        TrainingConfig("sac", learning_rate=5e-5)
    ) == pytest.approx(1e-4)
    assert train_module._warm_start_learning_rate(
        TrainingConfig("sac", warm_start_learning_rate=7e-5)
    ) == pytest.approx(7e-5)
    assert train_module.TOUCH_TEACHER_MODES == (
        "round_retention",
        "rigid_asymmetric",
        "slippery_retention",
        "fragile_balance",
    )
    assert TrainingConfig("ppo", warm_start_teacher="composed_touch").warm_start_teacher == (
        "composed_touch"
    )
    assert TrainingConfig("ppo").warm_start_validation_suite == "validation_procedural"
    assert TrainingConfig("ppo").warm_start_validation_limit == 24
    assert TrainingConfig("ppo").warm_start_min_safe_success_rate == pytest.approx(0.10)
    assert TrainingConfig("ppo").warm_start_policy_gate_suite == "validation_procedural"
    assert TrainingConfig("ppo").warm_start_policy_min_safe_success_rate == pytest.approx(0.10)
    assert TrainingConfig("sac").sac_policy_anchor_weight == pytest.approx(0.0)
    assert TrainingConfig("ppo").rl_regression_tolerance == pytest.approx(0.0)
    assert TrainingConfig("ppo").stop_on_rl_regression is False
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
    with pytest.raises(ValueError, match="validation suite"):
        TrainingConfig("ppo", warm_start_validation_suite="demo")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="validation_limit"):
        TrainingConfig("ppo", warm_start_validation_limit=0)
    with pytest.raises(ValueError, match="safe_success_rate"):
        TrainingConfig("ppo", warm_start_min_safe_success_rate=1.1)
    with pytest.raises(ValueError, match="warm_start_learning_rate"):
        TrainingConfig("ppo", warm_start_learning_rate=0.0)
    with pytest.raises(ValueError, match="policy gate suite"):
        TrainingConfig("ppo", warm_start_policy_gate_suite="demo")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="included in evaluation_suites"):
        TrainingConfig(
            "ppo",
            evaluation_suites=("test_procedural_holdout",),
            warm_start_policy_gate_suite="validation_procedural",
        )
    with pytest.raises(ValueError, match="policy_min_safe_success_rate"):
        TrainingConfig("ppo", warm_start_policy_min_safe_success_rate=1.1)
    with pytest.raises(ValueError, match="rl_regression_tolerance"):
        TrainingConfig("ppo", rl_regression_tolerance=-0.1)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig("ppo", learning_rate=0.0)
    with pytest.raises(ValueError, match="ppo_clip_range"):
        TrainingConfig("ppo", ppo_clip_range=0.0)
    with pytest.raises(ValueError, match="ppo_target_kl"):
        TrainingConfig("ppo", ppo_target_kl=0.0)
    with pytest.raises(ValueError, match="sac_learning_starts"):
        TrainingConfig("sac", sac_learning_starts=-1)
    with pytest.raises(ValueError, match="sac_ent_coef"):
        TrainingConfig("sac", sac_ent_coef=0.0)


def test_warm_start_teacher_validation_gate_uses_procedural_summary(monkeypatch) -> None:
    def fake_run_evaluation(controller_factory, suite, **kwargs):
        assert suite.name == "validation_procedural"
        assert len(suite.cases) == 2
        assert kwargs["controller_name"] == "safe_force_teacher"
        controller = controller_factory()
        observation = np.zeros(45, dtype=np.float32)
        controller.reset(observation, {"phase": "explore", "step": 0, "outcome": None})
        action = controller.act(observation, {"phase": "explore", "step": 0, "outcome": None})
        assert action.shape == (4,)
        return [
            {
                "object_family": "rounded",
                "outcome": "success",
                "safe_success": True,
                "peak_force": 0.30,
                "slip_events": 0,
                "final_lift_height": 0.050,
                "max_contacts": 3,
            },
            {
                "object_family": "fragile",
                "outcome": "damage",
                "safe_success": False,
                "peak_force": 0.90,
                "slip_events": 0,
                "final_lift_height": 0.020,
                "max_contacts": 3,
            },
        ]

    monkeypatch.setattr(train_module, "run_evaluation", fake_run_evaluation)
    config = TrainingConfig(
        "ppo",
        warm_start_teacher="safe_force",
        warm_start_validation_limit=2,
        warm_start_min_safe_success_rate=0.50,
    )

    summary = train_module.validate_warm_start_teacher(config)

    assert summary["safe_success_rate"] == pytest.approx(0.50)
    assert summary["failure_modes"] == {"damage": 1}
    with pytest.raises(RuntimeError, match="procedural validation gate"):
        train_module.validate_warm_start_teacher(
            TrainingConfig(
                "ppo",
                warm_start_teacher="safe_force",
                warm_start_validation_limit=2,
                warm_start_min_safe_success_rate=0.75,
            )
        )


def test_training_resets_rngs_before_building_environment(monkeypatch, tmp_path) -> None:
    calls: list[int] = []

    def fake_seed(seed: int) -> None:
        calls.append(seed)

    def fake_make_training_env(**kwargs):
        del kwargs
        assert calls == [17]
        raise RuntimeError("stop after seed")

    monkeypatch.setattr(train_module, "_seed_training_rngs", fake_seed)
    monkeypatch.setattr(train_module, "make_training_env", fake_make_training_env)

    with pytest.raises(RuntimeError, match="stop after seed"):
        train_module.train(
            TrainingConfig(
                "ppo",
                seed=17,
                output_root=tmp_path / "runs",
                checkpoint_root=tmp_path / "checkpoints",
                tensorboard_root=tmp_path / "tensorboard",
            )
        )


def test_training_rng_seed_is_reproducible() -> None:
    train_module._seed_training_rngs(23)
    python_value = random.random()
    numpy_value = float(np.random.random())

    train_module._seed_training_rngs(23)

    assert random.random() == pytest.approx(python_value)
    assert float(np.random.random()) == pytest.approx(numpy_value)


def test_warm_start_learning_rate_context_restores_optimizer() -> None:
    class DummyOptimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 1e-5}, {"lr": 2e-5}]

    optimizer = DummyOptimizer()
    with train_module._temporary_optimizer_learning_rate(optimizer, 3e-4):
        assert [group["lr"] for group in optimizer.param_groups] == [3e-4, 3e-4]

    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 2e-5]


def test_warm_start_policy_gate_uses_post_bc_checkpoint_summary() -> None:
    config = TrainingConfig(
        "ppo",
        warm_start_policy_min_safe_success_rate=0.50,
    )
    summaries = {
        "validation_procedural": {
            "safe_success_rate": 0.50,
            "failure_modes": {"damage": 1},
        }
    }

    summary = train_module.validate_warm_start_policy(summaries, config)

    assert summary["safe_success_rate"] == pytest.approx(0.50)
    with pytest.raises(RuntimeError, match="post-BC validation gate"):
        train_module.validate_warm_start_policy(
            {
                "validation_procedural": {
                    "safe_success_rate": 0.25,
                    "failure_modes": {"drop": 3},
                }
            },
            config,
        )


def test_checkpoint_preservation_report_compares_against_post_bc_score(tmp_path) -> None:
    config = TrainingConfig("sac", rl_regression_tolerance=0.125)

    preserved = train_module._checkpoint_preservation_record(
        config,
        baseline_label="step_0_warm_start",
        baseline_score=(0.50, -0.80),
        checkpoint_label="step_10000",
        checkpoint_score=(0.375, -0.70),
    )
    regressed = train_module._checkpoint_preservation_record(
        config,
        baseline_label="step_0_warm_start",
        baseline_score=(0.50, -0.80),
        checkpoint_label="step_20000",
        checkpoint_score=(0.25, -0.40),
    )

    assert preserved["passed"] is True
    assert preserved["safe_success_delta"] == pytest.approx(-0.125)
    assert preserved["required_safe_success_rate"] == pytest.approx(0.375)
    assert preserved["mean_peak_force_delta"] == pytest.approx(-0.10)
    assert regressed["passed"] is False
    report = tmp_path / "rl_preservation.jsonl"
    train_module._append_preservation_record(report, preserved)
    train_module._append_preservation_record(report, regressed)
    lines = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert [line["checkpoint"] for line in lines] == ["step_10000", "step_20000"]
    assert "regressed" in train_module._preservation_status_message("sac", regressed)


def test_checkpoint_evaluation_uses_all_suites_and_promotes_holdout(
    monkeypatch, tmp_path
) -> None:
    calls: list[str] = []

    def fake_evaluate_policy(
        model,
        suite,
        *,
        algorithm,
        checkpoint,
        history_length,
        output_prefix,
        env_config,
    ):
        del model, checkpoint, history_length, env_config
        calls.append(suite.name)
        assert algorithm == "ppo"
        assert len(suite.cases) == 1
        safe_success = suite.name == "test_procedural_holdout"
        return [
            {
                "object_family": "rounded",
                "outcome": "success" if safe_success else "timeout",
                "safe_success": safe_success,
                "peak_force": 0.25 if safe_success else 0.40,
                "slip_events": 0,
                "final_lift_height": 0.050 if safe_success else 0.0,
                "max_contacts": 3 if safe_success else 0,
            }
        ]

    monkeypatch.setattr(train_module, "evaluate_policy", fake_evaluate_policy)
    config = TrainingConfig("ppo", evaluation_limit=1)

    evaluation = train_module.evaluate_checkpoint(
        RecordingPolicy(),
        config,
        checkpoint=tmp_path / "checkpoint.zip",
        report_directory=tmp_path,
        step_label="step_1",
    )

    assert calls == ["validation_procedural", "test_procedural_holdout"]
    assert set(evaluation.records_by_suite) == set(config.evaluation_suites)
    assert evaluation.report_paths["validation_procedural"] == (
        tmp_path / "validation_procedural_step_1.csv"
    )
    assert evaluation.report_paths["test_procedural_holdout"] == (
        tmp_path / "test_procedural_holdout_step_1.csv"
    )
    summary = json.loads(evaluation.summary_path.read_text(encoding="utf-8"))
    assert evaluation.summaries == summary
    assert summary["validation_procedural"]["failure_modes"] == {"timeout_no_grip": 1}
    assert summary["test_procedural_holdout"]["safe_success_rate"] == pytest.approx(1.0)
    assert train_module._promotion_score(evaluation.records_by_suite, config) == (
        1.0,
        -0.25,
    )


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


def test_sac_policy_anchor_configuration_snapshots_post_bc_actor() -> None:
    class DummyAnchoredSac:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def set_policy_anchor(self, *, weight: float) -> None:
            self.calls.append(weight)

    model = DummyAnchoredSac()
    observations = np.zeros((3, 360), dtype=np.float32)
    actions = np.zeros((3, 4), dtype=np.float32)
    config = TrainingConfig(
        "sac",
        warm_start_transitions=3,
        sac_policy_anchor_weight=0.75,
    )

    train_module._configure_sac_warm_start_anchors(model, config, observations, actions)

    assert len(model.calls) == 1
    assert model.calls[0] == pytest.approx(0.75)


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
