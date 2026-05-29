"""PPO and SAC training entry point for BlindTouch.

This module keeps Stable-Baselines3 as an optional runtime dependency so the
simulation, rendering, and tests remain usable before training packages are
installed. Production learning is deliberately gated until the randomized
oracle feasibility diagnostic meets the readiness target recorded in todo.txt.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from .controllers import per_finger_max_taxel_force
from .env import BlindTouchEnv, EnvConfig, ObservationHistory
from .evaluate import (
    EvaluationSuite,
    build_locked_suite,
    run_evaluation,
    summarize_evaluation_records,
    write_csv_report,
    write_jsonl_report,
)
from .objects import SamplingConfig


AlgorithmName = Literal["ppo", "sac"]
CurriculumStage = Literal[
    "robust_upright", "upright", "fragile_upright", "all_poses", "with_chassis"
]
EvaluationSuiteName = Literal[
    "validation_procedural", "test_procedural_holdout", "test_pose", "test_stress"
]
WarmStartTeacherName = Literal["safe_force", "composed_touch"]
WarmStartProfileName = Literal["stage", "fragile_mix", "stratified_quality"]
WarmStartValidationSuiteName = EvaluationSuiteName
Observation = NDArray[np.float32]
Action = NDArray[np.float32]
POLICY_ENV_CONFIG = EnvConfig(max_episode_steps=140)
BASE_OBSERVATION_SIZE = 45
ALLOWED_EVALUATION_SUITES: tuple[EvaluationSuiteName, ...] = (
    "validation_procedural",
    "test_procedural_holdout",
    "test_pose",
    "test_stress",
)
DEFAULT_EVALUATION_SUITES: tuple[EvaluationSuiteName, ...] = (
    "validation_procedural",
    "test_procedural_holdout",
)
DEFAULT_PROMOTION_SUITE: EvaluationSuiteName = "test_procedural_holdout"
DEFAULT_WARM_START_LEARNING_RATES: dict[AlgorithmName, float] = {
    "ppo": 3e-4,
    "sac": 1e-4,
}
WARM_START_TARGET_FORCE = 0.34
WARM_START_FORCE_BAND = 0.10
WARM_START_CONTACT_THRESHOLD = 0.035
WARM_START_CLOSE_RATE = 0.18
WARM_START_TRIM_CLOSE_RATE = 0.08
WARM_START_RELEASE_RATE = 0.12
WARM_START_STABLE_WINDOW = 8
WARM_START_LATE_LIFT_STEP = 125
COMPOSED_TEACHER_PROBE_RATE = 0.26
COMPOSED_TEACHER_PROBE_STEPS = 74
COMPOSED_TEACHER_CONTACT_DWELL_STEPS = 2
COMPOSED_TEACHER_HIGH_FORCE = 0.42
COMPOSED_TEACHER_LATE_CONTACT_STEP = 57
COMPOSED_TEACHER_SPREAD_THRESHOLD = 0.34
COMPOSED_TEACHER_MANY_CONTACTS = 3
TOUCH_TEACHER_MODES = (
    "round_retention",
    "rigid_asymmetric",
    "slippery_retention",
    "fragile_balance",
)
FEASIBILITY_GATE_MESSAGE = (
    "Only the robust_upright and upright stage-1 curricula are currently certified for training. "
    "Use --allow-uncertified-environment only for deliberate pipeline dry runs "
    "on all_poses or with_chassis."
)
WARM_START_GATE_MESSAGE = (
    "Warm-start teacher did not meet the procedural validation gate; lower the gate only for "
    "deliberate debugging, or improve the teacher before behavior cloning."
)
WARM_START_POLICY_GATE_MESSAGE = (
    "Warm-start policy did not meet the post-BC validation gate; improve behavior cloning "
    "before spending compute on RL updates."
)


class PredictivePolicy(Protocol):
    """The small subset of an SB3 policy required by locked evaluation."""

    def predict(self, observation: Observation, deterministic: bool = True) -> tuple[Any, Any]:
        """Return one action for a stacked policy observation."""


@dataclass(frozen=True)
class TrainingConfig:
    """Auditable training configuration shared by PPO and SAC."""

    algorithm: AlgorithmName
    total_timesteps: int = 10_000
    seed: int = 0
    history_length: int = ObservationHistory.DEFAULT_HISTORY_LENGTH
    curriculum_stage: CurriculumStage = "upright"
    evaluation_frequency: int = 50_000
    evaluation_limit: int | None = None
    evaluation_suites: tuple[EvaluationSuiteName, ...] = DEFAULT_EVALUATION_SUITES
    promotion_suite: EvaluationSuiteName = DEFAULT_PROMOTION_SUITE
    output_root: Path = Path("runs")
    checkpoint_root: Path = Path("checkpoints")
    tensorboard_root: Path = Path("runs/tensorboard")
    device: str = "auto"
    allow_uncertified_environment: bool = False
    warm_start_transitions: int = 0
    warm_start_teacher: WarmStartTeacherName = "composed_touch"
    warm_start_profile: WarmStartProfileName = "stage"
    warm_start_epochs: int = 4
    warm_start_batch_size: int = 256
    warm_start_learning_rate: float | None = None
    warm_start_validation_suite: WarmStartValidationSuiteName = "validation_procedural"
    warm_start_validation_limit: int = 24
    warm_start_min_safe_success_rate: float = 0.10
    warm_start_policy_gate_suite: EvaluationSuiteName = "validation_procedural"
    warm_start_policy_min_safe_success_rate: float = 0.10
    sac_bc_anchor_weight: float = 0.0
    sac_bc_anchor_batch_size: int = 256
    sac_policy_anchor_weight: float = 0.0
    rl_regression_tolerance: float = 0.0
    stop_on_rl_regression: bool = False
    learning_rate: float | None = None
    ppo_clip_range: float | None = None
    ppo_target_kl: float | None = None
    sac_learning_starts: int | None = None
    sac_ent_coef: float | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in {"ppo", "sac"}:
            raise ValueError(f"Unsupported algorithm: {self.algorithm!r}")
        if self.total_timesteps < 0:
            raise ValueError("total_timesteps cannot be negative")
        if self.history_length < 1:
            raise ValueError("history_length must be positive")
        if self.curriculum_stage not in {
            "robust_upright",
            "upright",
            "fragile_upright",
            "all_poses",
            "with_chassis",
        }:
            raise ValueError(f"Unsupported curriculum stage: {self.curriculum_stage!r}")
        if self.evaluation_frequency < 1:
            raise ValueError("evaluation_frequency must be positive")
        if self.evaluation_limit is not None and self.evaluation_limit < 1:
            raise ValueError("evaluation_limit must be positive when provided")
        if not self.evaluation_suites:
            raise ValueError("evaluation_suites must contain at least one suite")
        if len(set(self.evaluation_suites)) != len(self.evaluation_suites):
            raise ValueError("evaluation_suites cannot contain duplicates")
        unsupported_suites = [
            suite for suite in self.evaluation_suites if suite not in ALLOWED_EVALUATION_SUITES
        ]
        if unsupported_suites:
            raise ValueError(f"Unsupported evaluation suites: {unsupported_suites!r}")
        if self.promotion_suite not in ALLOWED_EVALUATION_SUITES:
            raise ValueError(f"Unsupported promotion suite: {self.promotion_suite!r}")
        if self.promotion_suite not in self.evaluation_suites:
            raise ValueError("promotion_suite must be included in evaluation_suites")
        if self.warm_start_transitions < 0:
            raise ValueError("warm_start_transitions cannot be negative")
        if self.total_timesteps == 0 and self.warm_start_transitions == 0:
            raise ValueError("total_timesteps=0 requires warm_start_transitions")
        if self.warm_start_teacher not in {"safe_force", "composed_touch"}:
            raise ValueError(f"Unsupported warm-start teacher: {self.warm_start_teacher!r}")
        if self.warm_start_profile not in {"stage", "fragile_mix", "stratified_quality"}:
            raise ValueError(f"Unsupported warm-start profile: {self.warm_start_profile!r}")
        if self.warm_start_epochs < 1:
            raise ValueError("warm_start_epochs must be positive")
        if self.warm_start_batch_size < 1:
            raise ValueError("warm_start_batch_size must be positive")
        if self.warm_start_learning_rate is not None and self.warm_start_learning_rate <= 0.0:
            raise ValueError("warm_start_learning_rate must be positive when provided")
        if self.warm_start_validation_suite not in ALLOWED_EVALUATION_SUITES:
            raise ValueError(
                f"Unsupported warm-start validation suite: {self.warm_start_validation_suite!r}"
            )
        if self.warm_start_validation_limit < 1:
            raise ValueError("warm_start_validation_limit must be positive")
        if not 0.0 <= self.warm_start_min_safe_success_rate <= 1.0:
            raise ValueError("warm_start_min_safe_success_rate must be between 0 and 1")
        if self.warm_start_policy_gate_suite not in ALLOWED_EVALUATION_SUITES:
            raise ValueError(
                f"Unsupported warm-start policy gate suite: {self.warm_start_policy_gate_suite!r}"
            )
        if self.warm_start_policy_gate_suite not in self.evaluation_suites:
            raise ValueError("warm_start_policy_gate_suite must be included in evaluation_suites")
        if not 0.0 <= self.warm_start_policy_min_safe_success_rate <= 1.0:
            raise ValueError(
                "warm_start_policy_min_safe_success_rate must be between 0 and 1"
            )
        if self.sac_bc_anchor_weight < 0.0:
            raise ValueError("sac_bc_anchor_weight cannot be negative")
        if self.sac_bc_anchor_weight > 0.0 and self.warm_start_transitions < 1:
            raise ValueError("sac_bc_anchor_weight requires warm_start_transitions")
        if self.sac_bc_anchor_batch_size < 1:
            raise ValueError("sac_bc_anchor_batch_size must be positive")
        if self.sac_policy_anchor_weight < 0.0:
            raise ValueError("sac_policy_anchor_weight cannot be negative")
        if self.sac_policy_anchor_weight > 0.0 and self.warm_start_transitions < 1:
            raise ValueError("sac_policy_anchor_weight requires warm_start_transitions")
        if not 0.0 <= self.rl_regression_tolerance <= 1.0:
            raise ValueError("rl_regression_tolerance must be between 0 and 1")
        if self.learning_rate is not None and self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive when provided")
        if self.ppo_clip_range is not None and self.ppo_clip_range <= 0.0:
            raise ValueError("ppo_clip_range must be positive when provided")
        if self.ppo_target_kl is not None and self.ppo_target_kl <= 0.0:
            raise ValueError("ppo_target_kl must be positive when provided")
        if self.sac_learning_starts is not None and self.sac_learning_starts < 0:
            raise ValueError("sac_learning_starts cannot be negative")
        if self.sac_ent_coef is not None and self.sac_ent_coef <= 0.0:
            raise ValueError("sac_ent_coef must be positive when provided")

    @property
    def run_name(self) -> str:
        return f"{self.algorithm}_{self.curriculum_stage}_seed{self.seed}"


@dataclass(frozen=True)
class TrainingResult:
    """Paths and metrics resulting from one completed training invocation."""

    algorithm: AlgorithmName
    trained_timesteps: int
    final_checkpoint: Path
    best_checkpoint: Path
    latest_report: Path
    latest_reports: dict[str, Path]
    promotion_suite: str
    best_safe_success_rate: float


@dataclass(frozen=True)
class CheckpointEvaluation:
    """Evaluation reports and summaries for one saved checkpoint."""

    records_by_suite: dict[str, list[dict[str, Any]]]
    report_paths: dict[str, Path]
    summaries: dict[str, Any]
    summary_path: Path


@dataclass(frozen=True)
class _WarmStartDemoBucket:
    """One offline demonstration source for behavior-cloning warm starts."""

    name: str
    weight: float
    sampling_config: SamplingConfig
    accept_outcomes: tuple[str, ...] | None = None
    seed_offset: int = 17_000


class StackedPolicyController:
    """Adapt a learned stacked-observation policy to base evaluation episodes."""

    def __init__(
        self,
        model: PredictivePolicy,
        *,
        history_length: int = ObservationHistory.DEFAULT_HISTORY_LENGTH,
        deterministic: bool = True,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.model = model
        self.history_length = history_length
        self.deterministic = deterministic
        self._frames = np.zeros((history_length, 45), dtype=np.float32)
        self._first_action = True

    def reset(self, observation: Observation, info: Mapping[str, Any]) -> None:
        del info
        self._frames[:] = observation
        self._first_action = True

    def act(self, observation: Observation, info: Mapping[str, Any]) -> Action:
        del info
        if self._first_action:
            self._first_action = False
        else:
            self._frames[:-1] = self._frames[1:].copy()
            self._frames[-1] = observation
        action, _ = self.model.predict(
            self._frames.reshape(-1).copy(), deterministic=self.deterministic
        )
        return np.asarray(action, dtype=np.float32).reshape(4)


class _BudgetContinuation:
    """Close to a configured command budget, then lift."""

    def __init__(
        self,
        *,
        phases: tuple[tuple[float, int], ...],
        lift_rate: float,
        spent_budget: float,
    ) -> None:
        self.phases = phases
        self.lift_rate = lift_rate
        self.spent_budget = spent_budget
        self.total_budget = float(sum(abs(rate) * steps for rate, steps in phases))

    def act(self, observation: Observation) -> Action:
        del observation
        if self.spent_budget < self.total_budget:
            rate = self._current_close_rate()
            self.spent_budget += abs(rate)
            return np.array([0.0, rate, rate, rate], dtype=np.float32)
        return np.array([self.lift_rate, 0.0, 0.0, 0.0], dtype=np.float32)

    def _current_close_rate(self) -> float:
        cumulative = 0.0
        for rate, steps in self.phases:
            cumulative += abs(rate) * steps
            if self.spent_budget < cumulative:
                return rate
        return 0.0


class _SequenceContinuation:
    """Run one symmetric close segment, then lift."""

    def __init__(self, *, close_rate: float, close_steps: int, lift_rate: float) -> None:
        self.close_rate = close_rate
        self.close_steps = close_steps
        self.lift_rate = lift_rate
        self.step = 0

    def act(self, observation: Observation) -> Action:
        del observation
        if self.step < self.close_steps:
            self.step += 1
            return np.array(
                [0.0, self.close_rate, self.close_rate, self.close_rate],
                dtype=np.float32,
            )
        self.step += 1
        return np.array([self.lift_rate, 0.0, 0.0, 0.0], dtype=np.float32)


class _FragileBalancedContinuation:
    """Low-force per-finger balancing branch for fragile tactile modes."""

    def __init__(self) -> None:
        self.acquire_step = 0
        self.acquire_steps = 24

    def act(self, observation: Observation) -> Action:
        forces = per_finger_max_taxel_force(observation)
        if float(np.max(forces)) > 0.50:
            finger_actions = np.full(3, -0.02, dtype=np.float32)
        else:
            close_rate = 0.05 if self.acquire_step < self.acquire_steps else 0.04
            finger_actions = np.where(
                forces < 0.46,
                close_rate,
                np.where(forces > 0.50, -0.02, 0.0),
            ).astype(np.float32)
        if self.acquire_step < self.acquire_steps:
            self.acquire_step += 1
            return np.r_[0.0, finger_actions].astype(np.float32)
        return np.r_[0.32, finger_actions].astype(np.float32)


class _ComposedTouchTeacher:
    """Touch-only mode teacher used for behavior-cloning warm starts.

    The branch names are mnemonic labels for action families; this teacher does
    not read object names, masses, friction, safe-force limits, or diagnostics.
    """

    def __init__(self) -> None:
        self.step = 0
        self.first_contact_step: int | None = None
        self.first_two_contact_step: int | None = None
        self.first_three_contact_step: int | None = None
        self.max_force_seen = 0.0
        self.max_contact_count_seen = 0
        self.final_forces = np.zeros(3, dtype=np.float32)
        self.selected_branch: str | None = None
        self.continuation: Any | None = None

    def act(self, stacked_observation: Observation) -> Action:
        observation = _latest_frame(stacked_observation)
        self._update_touch_history(observation)
        if self.continuation is None:
            if not self._probe_complete():
                self.step += 1
                rate = COMPOSED_TEACHER_PROBE_RATE
                return np.array([0.0, rate, rate, rate], dtype=np.float32)
            self.selected_branch = self._select_branch()
            self.continuation = self._make_continuation()

        self.step += 1
        return self.continuation.act(observation)

    def _make_continuation(self) -> Any:
        spent_budget = COMPOSED_TEACHER_PROBE_RATE * self.step
        if self.selected_branch == "round_retention":
            return _BudgetContinuation(
                phases=((0.26, 63),),
                lift_rate=0.70,
                spent_budget=spent_budget,
            )
        if self.selected_branch == "rigid_asymmetric":
            return _BudgetContinuation(
                phases=((0.20, 80),),
                lift_rate=0.70,
                spent_budget=spent_budget,
            )
        if self.selected_branch == "slippery_retention":
            return _SequenceContinuation(close_rate=0.10, close_steps=24, lift_rate=0.35)
        if self.selected_branch == "fragile_balance":
            return _FragileBalancedContinuation()
        raise AssertionError(self.selected_branch)

    def _select_branch(self) -> str:
        if self.first_contact_step is None:
            return "round_retention"
        first_contact = self.first_contact_step
        final_contacts = int(np.count_nonzero(self.final_forces >= WARM_START_CONTACT_THRESHOLD))
        force_spread = float(np.ptp(self.final_forces))
        if first_contact >= COMPOSED_TEACHER_LATE_CONTACT_STEP:
            if (
                self.max_force_seen >= COMPOSED_TEACHER_HIGH_FORCE
                or final_contacts >= COMPOSED_TEACHER_MANY_CONTACTS
            ):
                return "round_retention"
            return "fragile_balance"
        if (
            self.max_force_seen >= COMPOSED_TEACHER_HIGH_FORCE
            or force_spread >= COMPOSED_TEACHER_SPREAD_THRESHOLD
        ):
            return "rigid_asymmetric"
        return "slippery_retention"

    def _probe_complete(self) -> bool:
        if self.step >= COMPOSED_TEACHER_PROBE_STEPS:
            return True
        if self.first_contact_step is None:
            return False
        return self.step >= self.first_contact_step + COMPOSED_TEACHER_CONTACT_DWELL_STEPS

    def _update_touch_history(self, observation: Observation) -> None:
        forces = per_finger_max_taxel_force(observation)
        contact_count = int(np.count_nonzero(forces >= WARM_START_CONTACT_THRESHOLD))
        self.final_forces = forces
        self.max_force_seen = max(self.max_force_seen, float(np.max(forces)))
        self.max_contact_count_seen = max(self.max_contact_count_seen, contact_count)
        if contact_count >= 1 and self.first_contact_step is None:
            self.first_contact_step = self.step
        if contact_count >= 2 and self.first_two_contact_step is None:
            self.first_two_contact_step = self.step
        if contact_count >= 3 and self.first_three_contact_step is None:
            self.first_three_contact_step = self.step


class WarmStartTeacherController:
    """Evaluate warm-start teachers with the same stacked history used for BC."""

    def __init__(
        self,
        teacher_name: WarmStartTeacherName,
        *,
        history_length: int = ObservationHistory.DEFAULT_HISTORY_LENGTH,
    ) -> None:
        if teacher_name not in {"safe_force", "composed_touch"}:
            raise ValueError(f"Unsupported warm-start teacher: {teacher_name!r}")
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.teacher_name = teacher_name
        self.history_length = history_length
        self._frames = np.zeros((history_length, BASE_OBSERVATION_SIZE), dtype=np.float32)
        self._composed_teacher = _ComposedTouchTeacher()
        self._first_action = True

    def reset(self, observation: Observation, info: Mapping[str, Any]) -> None:
        del info
        self._frames[:] = observation
        self._composed_teacher = _ComposedTouchTeacher()
        self._first_action = True

    def act(self, observation: Observation, info: Mapping[str, Any]) -> Action:
        del info
        if self._first_action:
            self._first_action = False
        else:
            self._frames[:-1] = self._frames[1:].copy()
            self._frames[-1] = observation
        stacked_observation = self._frames.reshape(-1).copy()
        if self.teacher_name == "safe_force":
            return _history_safe_force_teacher_action(stacked_observation)
        if self.teacher_name == "composed_touch":
            return self._composed_teacher.act(stacked_observation)
        raise AssertionError(self.teacher_name)


def algorithm_hyperparameters(
    algorithm: AlgorithmName, config: TrainingConfig | None = None
) -> dict[str, Any]:
    """Return the fixed first-pass model settings from the project roadmap."""

    common: dict[str, Any] = {
        "batch_size": 256,
        "gamma": 0.99,
        "policy_kwargs": {"net_arch": [256, 256]},
    }
    if algorithm == "ppo":
        hyperparameters: dict[str, Any] = {
            **common,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "gae_lambda": 0.95,
        }
        if config is not None and config.learning_rate is not None:
            hyperparameters["learning_rate"] = config.learning_rate
        if config is not None and config.ppo_clip_range is not None:
            hyperparameters["clip_range"] = config.ppo_clip_range
        if config is not None and config.ppo_target_kl is not None:
            hyperparameters["target_kl"] = config.ppo_target_kl
        return hyperparameters
    if algorithm == "sac":
        hyperparameters = {
            **common,
            "learning_rate": 1e-4,
            "buffer_size": 1_000_000,
            "learning_starts": 10_000,
            "ent_coef": "auto",
        }
        if config is not None and config.learning_rate is not None:
            hyperparameters["learning_rate"] = config.learning_rate
        if config is not None and config.sac_learning_starts is not None:
            hyperparameters["learning_starts"] = config.sac_learning_starts
        if config is not None and config.sac_ent_coef is not None:
            hyperparameters["ent_coef"] = config.sac_ent_coef
        return hyperparameters
    raise ValueError(f"Unsupported algorithm: {algorithm!r}")


def make_training_env(
    *,
    history_length: int = ObservationHistory.DEFAULT_HISTORY_LENGTH,
    env_config: EnvConfig = POLICY_ENV_CONFIG,
    curriculum_stage: CurriculumStage = "upright",
    sampling_config: SamplingConfig | None = None,
) -> ObservationHistory:
    """Create the touch-only 360-value policy environment used by both algorithms."""

    selected_sampling = sampling_config or curriculum_sampling_config(curriculum_stage)
    return ObservationHistory(
        BlindTouchEnv(config=env_config, sampling_config=selected_sampling),
        history_length=history_length,
    )


def curriculum_sampling_config(stage: CurriculumStage) -> SamplingConfig:
    """Map a curriculum stage to abstract training families and stable poses."""

    if stage == "robust_upright":
        return SamplingConfig(
            training_families=("rounded", "container"),
            allowed_poses=("upright",),
            offset_range=(-0.001, 0.001),
            friction_range=(0.65, 1.20),
            mass_range=(0.030, 0.110),
            nominal_pad_force_capacity=0.55,
            holding_force_margin=0.70,
            safe_force_margin=4.0,
        )
    if stage == "upright":
        return SamplingConfig(
            training_families=("rounded", "container", "package", "slippery", "fragile"),
            allowed_poses=("upright",),
            offset_range=(-0.002, 0.002),
            safe_force_margin=3.0,
        )
    if stage == "fragile_upright":
        return SamplingConfig(
            training_families=("fragile", "rounded", "container", "package"),
            allowed_poses=("upright",),
            offset_range=(-0.002, 0.002),
            friction_range=(0.45, 1.10),
            mass_range=(0.040, 0.140),
            safe_force_headroom_range=(2.60, 3.80),
            nominal_pad_force_capacity=0.55,
            holding_force_margin=0.75,
        )
    if stage == "all_poses":
        return SamplingConfig(
            training_families=("rounded", "container", "package", "slippery", "fragile")
        )
    if stage == "with_chassis":
        return SamplingConfig()
    raise ValueError(f"Unsupported curriculum stage: {stage!r}")


def require_feasibility_acknowledgement(
    curriculum_stage: CurriculumStage, allow_uncertified_environment: bool
) -> None:
    """Refuse accidental learning on curriculum stages not yet certified."""

    if curriculum_stage not in {"robust_upright", "upright"} and not allow_uncertified_environment:
        raise RuntimeError(FEASIBILITY_GATE_MESSAGE)


def evaluate_policy(
    model: PredictivePolicy,
    suite: EvaluationSuite | str,
    *,
    algorithm: AlgorithmName,
    checkpoint: str,
    history_length: int = ObservationHistory.DEFAULT_HISTORY_LENGTH,
    output_prefix: Path | None = None,
    env_config: EnvConfig = POLICY_ENV_CONFIG,
) -> list[dict[str, Any]]:
    """Evaluate a learned model using the same observation history as training."""

    records = run_evaluation(
        lambda: StackedPolicyController(model, history_length=history_length),
        suite,
        controller_name=algorithm,
        checkpoint=checkpoint,
        env_config=env_config,
    )
    if output_prefix is not None:
        write_csv_report(records, output_prefix.with_suffix(".csv"))
        write_jsonl_report(records, output_prefix.with_suffix(".jsonl"))
    return records


def validate_warm_start_teacher(config: TrainingConfig) -> dict[str, Any]:
    """Run the scripted teacher through a locked procedural gate before BC."""

    suite = build_locked_suite(
        config.warm_start_validation_suite,
        limit=config.warm_start_validation_limit,
    )
    records = run_evaluation(
        lambda: WarmStartTeacherController(
            config.warm_start_teacher, history_length=config.history_length
        ),
        suite,
        controller_name=f"{config.warm_start_teacher}_teacher",
        env_config=POLICY_ENV_CONFIG,
    )
    summary = summarize_evaluation_records(records)
    safe_success_rate = float(summary["safe_success_rate"])
    if safe_success_rate < config.warm_start_min_safe_success_rate:
        raise RuntimeError(
            f"{WARM_START_GATE_MESSAGE} "
            f"teacher={config.warm_start_teacher}, "
            f"suite={config.warm_start_validation_suite}, "
            f"safe_success_rate={safe_success_rate:.3f}, "
            f"required={config.warm_start_min_safe_success_rate:.3f}, "
            f"failure_modes={summary['failure_modes']}"
        )
    return summary


def validate_warm_start_policy(
    summaries_by_suite: Mapping[str, Mapping[str, Any]], config: TrainingConfig
) -> Mapping[str, Any]:
    """Gate the cloned policy before allowing RL to update it."""

    summary = summaries_by_suite[config.warm_start_policy_gate_suite]
    safe_success_rate = float(summary["safe_success_rate"])
    if safe_success_rate < config.warm_start_policy_min_safe_success_rate:
        raise RuntimeError(
            f"{WARM_START_POLICY_GATE_MESSAGE} "
            f"algorithm={config.algorithm}, "
            f"suite={config.warm_start_policy_gate_suite}, "
            f"safe_success_rate={safe_success_rate:.3f}, "
            f"required={config.warm_start_policy_min_safe_success_rate:.3f}, "
            f"failure_modes={summary['failure_modes']}"
        )
    return summary


def build_model(config: TrainingConfig, env: ObservationHistory) -> Any:
    """Instantiate an SB3 PPO or SAC model, importing training dependencies lazily."""

    algorithms = _load_algorithms()
    model_class = algorithms[config.algorithm]
    return model_class(
        "MlpPolicy",
        env,
        seed=config.seed,
        verbose=1,
        tensorboard_log=str(config.tensorboard_root),
        device=config.device,
        **algorithm_hyperparameters(config.algorithm, config),
    )


def train(config: TrainingConfig) -> TrainingResult:
    """Train one algorithm, checkpoint it, and evaluate fixed validation cases."""

    require_feasibility_acknowledgement(
        config.curriculum_stage, config.allow_uncertified_environment
    )
    _seed_training_rngs(config.seed)
    run_directory = config.output_root / config.run_name
    checkpoint_directory = config.checkpoint_root / config.run_name
    report_directory = run_directory / "evaluation"
    for directory in (run_directory, checkpoint_directory, report_directory, config.tensorboard_root):
        directory.mkdir(parents=True, exist_ok=True)
    _write_training_config(config, run_directory / "config.json")

    env = make_training_env(
        history_length=config.history_length, curriculum_stage=config.curriculum_stage
    )
    model_class = _load_algorithms()[config.algorithm]
    model = build_model(config, env)
    if config.warm_start_transitions > 0:
        loss = warm_start_from_safe_force_controller(model, config)
        print(
            f"{config.algorithm.upper()} {config.warm_start_teacher} warm start: "
            f"{config.warm_start_transitions} transitions, final_loss={loss:.4f}"
        )
    best_score: tuple[float, float] | None = None
    best_checkpoint = checkpoint_directory / "best.zip"
    latest_reports: dict[str, Path] = {}
    latest_report = report_directory / f"{config.promotion_suite}_step_0.csv"
    next_evaluation = min(config.evaluation_frequency, config.total_timesteps)
    final_checkpoint: Path | None = None
    warm_start_score: tuple[float, float] | None = None
    preservation_report = report_directory / "rl_preservation.jsonl"
    try:
        if config.warm_start_transitions > 0:
            checkpoint_base = checkpoint_directory / "step_0_warm_start"
            model.save(str(checkpoint_base))
            checkpoint = checkpoint_base.with_suffix(".zip")
            loaded_model = model_class.load(str(checkpoint), device=config.device)
            evaluation = evaluate_checkpoint(
                loaded_model,
                config,
                checkpoint=checkpoint,
                report_directory=report_directory,
                step_label="step_0_warm_start",
            )
            latest_reports = evaluation.report_paths
            latest_report = latest_reports[config.promotion_suite]
            best_score = _promotion_score(evaluation.records_by_suite, config)
            warm_start_score = best_score
            if config.total_timesteps > 0:
                validate_warm_start_policy(evaluation.summaries, config)
                _append_preservation_record(
                    preservation_report,
                    _checkpoint_preservation_record(
                        config,
                        baseline_label="step_0_warm_start",
                        baseline_score=warm_start_score,
                        checkpoint_label="step_0_warm_start",
                        checkpoint_score=warm_start_score,
                    ),
                )
            shutil.copyfile(checkpoint, best_checkpoint)
            final_checkpoint = checkpoint
        while model.num_timesteps < config.total_timesteps:
            remaining = next_evaluation - model.num_timesteps
            model.learn(total_timesteps=max(1, remaining), reset_num_timesteps=False)
            step = int(model.num_timesteps)
            checkpoint_base = checkpoint_directory / f"step_{step}"
            model.save(str(checkpoint_base))
            checkpoint = checkpoint_base.with_suffix(".zip")
            final_checkpoint = checkpoint
            loaded_model = model_class.load(str(checkpoint), device=config.device)
            evaluation = evaluate_checkpoint(
                loaded_model,
                config,
                checkpoint=checkpoint,
                report_directory=report_directory,
                step_label=f"step_{step}",
            )
            latest_reports = evaluation.report_paths
            latest_report = latest_reports[config.promotion_suite]
            score = _promotion_score(evaluation.records_by_suite, config)
            if best_score is None or score > best_score:
                shutil.copyfile(checkpoint, best_checkpoint)
                best_score = score
            if warm_start_score is not None:
                preservation = _checkpoint_preservation_record(
                    config,
                    baseline_label="step_0_warm_start",
                    baseline_score=warm_start_score,
                    checkpoint_label=f"step_{step}",
                    checkpoint_score=score,
                )
                _append_preservation_record(preservation_report, preservation)
                print(_preservation_status_message(config.algorithm, preservation))
                if config.stop_on_rl_regression and not preservation["passed"]:
                    break
            if model.num_timesteps >= config.total_timesteps:
                break
            next_evaluation = min(
                config.total_timesteps, next_evaluation + config.evaluation_frequency
            )
    finally:
        env.close()

    if best_score is None:
        raise RuntimeError("Training completed without evaluating a checkpoint")
    if final_checkpoint is None:
        raise RuntimeError("Training completed without saving a checkpoint")
    return TrainingResult(
        algorithm=config.algorithm,
        trained_timesteps=int(model.num_timesteps),
        final_checkpoint=final_checkpoint,
        best_checkpoint=best_checkpoint,
        latest_report=latest_report,
        latest_reports=latest_reports,
        promotion_suite=config.promotion_suite,
        best_safe_success_rate=best_score[0],
    )


def _seed_training_rngs(seed: int) -> None:
    """Reset process-level RNGs so repeated warm-start runs are comparable."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as th
    except ImportError:
        return

    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def _snapshot_torch_state_dict(module: Any) -> dict[str, Any]:
    """Clone a torch module state dict without copying autograd history."""

    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


def evaluate_checkpoint(
    model: PredictivePolicy,
    config: TrainingConfig,
    *,
    checkpoint: Path,
    report_directory: Path,
    step_label: str,
) -> CheckpointEvaluation:
    """Evaluate one checkpoint on every configured suite and write summaries."""

    records_by_suite: dict[str, list[dict[str, Any]]] = {}
    report_paths: dict[str, Path] = {}
    summaries: dict[str, Any] = {}
    for suite_name in config.evaluation_suites:
        suite = build_locked_suite(suite_name, limit=config.evaluation_limit)
        prefix = report_directory / f"{suite_name}_{step_label}"
        records = evaluate_policy(
            model,
            suite,
            algorithm=config.algorithm,
            checkpoint=str(checkpoint),
            history_length=config.history_length,
            output_prefix=prefix,
            env_config=POLICY_ENV_CONFIG,
        )
        records_by_suite[suite_name] = records
        report_paths[suite_name] = prefix.with_suffix(".csv")
        summaries[suite_name] = summarize_evaluation_records(records)

    summary_path = report_directory / f"summary_{step_label}.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CheckpointEvaluation(records_by_suite, report_paths, summaries, summary_path)


def _checkpoint_preservation_record(
    config: TrainingConfig,
    *,
    baseline_label: str,
    baseline_score: tuple[float, float],
    checkpoint_label: str,
    checkpoint_score: tuple[float, float],
) -> dict[str, Any]:
    """Describe whether an RL checkpoint preserved the post-BC safe-success rate."""

    baseline_safe_success_rate = baseline_score[0]
    safe_success_rate = checkpoint_score[0]
    baseline_mean_peak_force = -baseline_score[1]
    mean_peak_force = -checkpoint_score[1]
    required_safe_success_rate = max(
        0.0, baseline_safe_success_rate - config.rl_regression_tolerance
    )
    return {
        "suite": config.promotion_suite,
        "baseline_checkpoint": baseline_label,
        "checkpoint": checkpoint_label,
        "baseline_safe_success_rate": baseline_safe_success_rate,
        "safe_success_rate": safe_success_rate,
        "safe_success_delta": safe_success_rate - baseline_safe_success_rate,
        "required_safe_success_rate": required_safe_success_rate,
        "passed": safe_success_rate >= required_safe_success_rate,
        "baseline_mean_peak_force": baseline_mean_peak_force,
        "mean_peak_force": mean_peak_force,
        "mean_peak_force_delta": mean_peak_force - baseline_mean_peak_force,
    }


def _append_preservation_record(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def _preservation_status_message(algorithm: AlgorithmName, record: Mapping[str, Any]) -> str:
    status = "preserved" if record["passed"] else "regressed"
    return (
        f"{algorithm.upper()} preservation {record['checkpoint']}: {status}; "
        f"{record['suite']} safe-success={record['safe_success_rate']:.3f} "
        f"vs post-BC {record['baseline_safe_success_rate']:.3f}"
    )


def _promotion_score(
    records_by_suite: Mapping[str, list[dict[str, Any]]], config: TrainingConfig
) -> tuple[float, float]:
    return _policy_score(records_by_suite[config.promotion_suite])


def _policy_score(records: list[dict[str, Any]]) -> tuple[float, float]:
    success_rate = sum(bool(record["safe_success"]) for record in records) / len(records)
    mean_peak_force = float(np.mean([record["peak_force"] for record in records]))
    return success_rate, -mean_peak_force


def warm_start_from_safe_force_controller(model: Any, config: TrainingConfig) -> float:
    """Behavior-clone a small tactile force-regulation prior into the actor."""

    validate_warm_start_teacher(config)
    observations, actions = _collect_warm_start_demonstrations(config)
    try:
        import torch as th
    except ImportError as error:
        raise RuntimeError("Scripted warm start requires PyTorch from Stable-Baselines3") from error

    obs_tensor = th.as_tensor(observations, dtype=th.float32, device=model.device)
    action_tensor = th.as_tensor(actions, dtype=th.float32, device=model.device)
    optimizer = _warm_start_optimizer(model, config.algorithm)
    rng = np.random.default_rng(config.seed + 29_000)
    final_loss = 0.0
    with _temporary_optimizer_learning_rate(
        optimizer, _warm_start_learning_rate(config)
    ):
        for _ in range(config.warm_start_epochs):
            for indices in _batch_indices(
                rng, len(observations), config.warm_start_batch_size
            ):
                batch_obs = obs_tensor[indices]
                batch_actions = action_tensor[indices]
                if config.algorithm == "ppo":
                    predicted_actions = model.policy.get_distribution(batch_obs).mode()
                else:
                    predicted_actions = model.actor(batch_obs, deterministic=True)
                loss = (predicted_actions - batch_actions).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu().item())
    _configure_sac_warm_start_anchors(model, config, observations, actions)
    return final_loss


def _warm_start_optimizer(model: Any, algorithm: AlgorithmName) -> Any:
    if algorithm == "ppo":
        return model.policy.optimizer
    return model.actor.optimizer


def _warm_start_learning_rate(config: TrainingConfig) -> float:
    return (
        config.warm_start_learning_rate
        if config.warm_start_learning_rate is not None
        else DEFAULT_WARM_START_LEARNING_RATES[config.algorithm]
    )


@contextmanager
def _temporary_optimizer_learning_rate(optimizer: Any, learning_rate: float):
    original_learning_rates = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    try:
        yield
    finally:
        for group, original_learning_rate in zip(
            optimizer.param_groups, original_learning_rates, strict=True
        ):
            group["lr"] = original_learning_rate


def _configure_sac_warm_start_anchors(
    model: Any,
    config: TrainingConfig,
    observations: NDArray[np.float32],
    actions: NDArray[np.float32],
) -> None:
    """Attach post-BC preservation anchors for SAC actor updates."""

    if config.algorithm != "sac":
        return
    if config.sac_bc_anchor_weight > 0.0:
        if not hasattr(model, "set_bc_anchor"):
            raise RuntimeError("SAC BC anchor requires the anchored SAC training class")
        model.set_bc_anchor(
            observations,
            actions,
            weight=config.sac_bc_anchor_weight,
            batch_size=config.sac_bc_anchor_batch_size,
            seed=config.seed + 41_000,
        )
    if config.sac_policy_anchor_weight > 0.0:
        if not hasattr(model, "set_policy_anchor"):
            raise RuntimeError("SAC policy anchor requires the anchored SAC training class")
        model.set_policy_anchor(weight=config.sac_policy_anchor_weight)


def _configure_sac_bc_anchor(
    model: Any,
    config: TrainingConfig,
    observations: NDArray[np.float32],
    actions: NDArray[np.float32],
) -> None:
    """Backward-compatible wrapper for tests and older scratch tooling."""

    _configure_sac_warm_start_anchors(model, config, observations, actions)


def _collect_warm_start_demonstrations(
    config: TrainingConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if config.warm_start_teacher == "safe_force":
        return _collect_safe_force_demonstrations(config)
    if config.warm_start_teacher == "composed_touch":
        return _collect_composed_touch_demonstrations(config)
    raise ValueError(f"Unsupported warm-start teacher: {config.warm_start_teacher!r}")


def _collect_safe_force_demonstrations(
    config: TrainingConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    observations: list[Observation] = []
    actions: list[Action] = []
    stages = _warm_start_curriculum_stages(config)
    envs = {
        stage: make_training_env(history_length=config.history_length, curriculum_stage=stage)
        for stage in stages
    }
    stage_index = 0
    seed = config.seed + 17_000
    try:
        stacked_observation, _ = envs[stages[stage_index]].reset(seed=seed)
        while len(observations) < config.warm_start_transitions:
            action = _history_safe_force_teacher_action(stacked_observation)
            observations.append(stacked_observation.copy())
            actions.append(action.copy())
            stacked_observation, _, terminated, truncated, _ = envs[stages[stage_index]].step(action)
            if terminated or truncated:
                seed += 1
                stage_index = (stage_index + 1) % len(stages)
                stacked_observation, _ = envs[stages[stage_index]].reset(seed=seed)
    finally:
        for env in envs.values():
            env.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def _collect_composed_touch_demonstrations(
    config: TrainingConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if config.warm_start_profile == "stratified_quality":
        return _collect_stratified_quality_demonstrations(config)

    observations: list[Observation] = []
    actions: list[Action] = []
    stages = _warm_start_curriculum_stages(config)
    envs = {
        stage: make_training_env(history_length=config.history_length, curriculum_stage=stage)
        for stage in stages
    }
    stage_index = 0
    seed = config.seed + 17_000
    teacher = _ComposedTouchTeacher()
    try:
        stacked_observation, _ = envs[stages[stage_index]].reset(seed=seed)
        while len(observations) < config.warm_start_transitions:
            action = teacher.act(stacked_observation)
            observations.append(stacked_observation.copy())
            actions.append(action.copy())
            stacked_observation, _, terminated, truncated, _ = envs[stages[stage_index]].step(action)
            if terminated or truncated:
                seed += 1
                stage_index = (stage_index + 1) % len(stages)
                teacher = _ComposedTouchTeacher()
                stacked_observation, _ = envs[stages[stage_index]].reset(seed=seed)
    finally:
        for env in envs.values():
            env.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def _collect_stratified_quality_demonstrations(
    config: TrainingConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    buckets = _stratified_quality_demo_buckets(config)
    quotas = _transition_quotas(config.warm_start_transitions, buckets)
    observation_chunks: list[NDArray[np.float32]] = []
    action_chunks: list[NDArray[np.float32]] = []
    stats: list[dict[str, Any]] = []
    for bucket in buckets:
        observations, actions, bucket_stats = _collect_demo_bucket(
            bucket,
            target_transitions=quotas[bucket.name],
            seed=config.seed + bucket.seed_offset,
            history_length=config.history_length,
        )
        observation_chunks.append(observations)
        action_chunks.append(actions)
        stats.append(bucket_stats)

    print("WARM_START_DEMO_STATS " + json.dumps(stats, sort_keys=True), flush=True)
    return (
        np.concatenate(observation_chunks, axis=0).astype(np.float32),
        np.concatenate(action_chunks, axis=0).astype(np.float32),
    )


def _collect_demo_bucket(
    bucket: _WarmStartDemoBucket,
    *,
    target_transitions: int,
    seed: int,
    history_length: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]]:
    if target_transitions == 0:
        return (
            np.empty((0, history_length * BASE_OBSERVATION_SIZE), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            {
                "bucket": bucket.name,
                "target_transitions": 0,
                "collected_transitions": 0,
                "attempts": 0,
                "accepted_episodes": 0,
                "accept_outcomes": (
                    list(bucket.accept_outcomes) if bucket.accept_outcomes else "all"
                ),
                "outcomes": {},
                "accepted_outcomes": {},
                "accepted_families": {},
                "accepted_poses": {},
            },
        )

    observations: list[Observation] = []
    actions: list[Action] = []
    attempts = 0
    accepted = 0
    outcomes: dict[str, int] = {}
    accepted_outcomes: dict[str, int] = {}
    accepted_families: dict[str, int] = {}
    accepted_poses: dict[str, int] = {}
    max_attempts = max(500, target_transitions)
    env = ObservationHistory(
        BlindTouchEnv(config=POLICY_ENV_CONFIG, sampling_config=bucket.sampling_config),
        history_length=history_length,
    )
    try:
        while len(observations) < target_transitions and attempts < max_attempts:
            stacked_observation, _ = env.reset(seed=seed + attempts)
            teacher = _ComposedTouchTeacher()
            episode_observations: list[Observation] = []
            episode_actions: list[Action] = []
            while True:
                action = teacher.act(stacked_observation)
                episode_observations.append(stacked_observation.copy())
                episode_actions.append(action.copy())
                stacked_observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            attempts += 1
            outcome = str(info["outcome"])
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if bucket.accept_outcomes is not None and outcome not in bucket.accept_outcomes:
                continue

            accepted += 1
            accepted_outcomes[outcome] = accepted_outcomes.get(outcome, 0) + 1
            episode_object = env.unwrapped.object_params
            if episode_object is not None:
                accepted_families[episode_object.family] = (
                    accepted_families.get(episode_object.family, 0) + 1
                )
                accepted_poses[episode_object.pose] = (
                    accepted_poses.get(episode_object.pose, 0) + 1
                )
            observations.extend(episode_observations)
            actions.extend(episode_actions)
    finally:
        env.close()

    if len(observations) < target_transitions:
        raise RuntimeError(
            f"{bucket.name} collected only {len(observations)}/{target_transitions} "
            f"transitions after {attempts} attempts; outcomes={outcomes}; "
            f"accepted_outcomes={accepted_outcomes}"
        )

    stats = {
        "bucket": bucket.name,
        "target_transitions": target_transitions,
        "collected_transitions": target_transitions,
        "attempts": attempts,
        "accepted_episodes": accepted,
        "accept_outcomes": list(bucket.accept_outcomes) if bucket.accept_outcomes else "all",
        "outcomes": outcomes,
        "accepted_outcomes": accepted_outcomes,
        "accepted_families": accepted_families,
        "accepted_poses": accepted_poses,
    }
    return (
        np.asarray(observations[:target_transitions], dtype=np.float32),
        np.asarray(actions[:target_transitions], dtype=np.float32),
        stats,
    )


def _stratified_quality_demo_buckets(
    config: TrainingConfig,
) -> tuple[_WarmStartDemoBucket, ...]:
    success = ("success",)
    return (
        _WarmStartDemoBucket(
            "stage_core",
            0.95,
            curriculum_sampling_config(config.curriculum_stage),
        ),
        _WarmStartDemoBucket(
            "fragile_low_margin_success",
            0.017,
            SamplingConfig(
                training_families=("fragile",),
                allowed_poses=("upright", "side_x", "side_y"),
                offset_range=(-0.004, 0.004),
                friction_range=(0.65, 1.20),
                mass_range=(0.025, 0.085),
                safe_force_headroom_range=(2.0, 3.1),
                nominal_pad_force_capacity=0.55,
                holding_force_margin=0.75,
            ),
            accept_outcomes=success,
            seed_offset=43_000,
        ),
        _WarmStartDemoBucket(
            "slippery_gap_success",
            0.017,
            SamplingConfig(
                training_families=("slippery",),
                allowed_poses=("upright", "side_x", "side_y"),
                offset_range=(-0.004, 0.004),
                friction_range=(0.16, 0.32),
                mass_range=(0.030, 0.120),
                safe_force_margin=2.4,
                nominal_pad_force_capacity=0.75,
                holding_force_margin=0.85,
            ),
            accept_outcomes=success,
            seed_offset=53_000,
        ),
        _WarmStartDemoBucket(
            "rigid_side_gap_success",
            0.016,
            SamplingConfig(
                training_families=("container", "package", "chassis"),
                allowed_poses=("upright", "side_x", "side_y", "wheels_down"),
                offset_range=(-0.004, 0.004),
                friction_range=(0.30, 1.20),
                mass_range=(0.040, 0.180),
                safe_force_margin=2.5,
                nominal_pad_force_capacity=0.65,
                holding_force_margin=0.80,
            ),
            accept_outcomes=success,
            seed_offset=63_000,
        ),
    )


def _transition_quotas(
    total: int, buckets: tuple[_WarmStartDemoBucket, ...]
) -> dict[str, int]:
    quotas = {bucket.name: int(total * bucket.weight) for bucket in buckets}
    quotas[buckets[0].name] += total - sum(quotas.values())
    return quotas


def _warm_start_curriculum_stages(config: TrainingConfig) -> tuple[CurriculumStage, ...]:
    if config.warm_start_profile == "stage":
        return (config.curriculum_stage,)
    if config.warm_start_profile == "fragile_mix":
        return tuple(dict.fromkeys((config.curriculum_stage, "fragile_upright")))
    if config.warm_start_profile == "stratified_quality":
        return (config.curriculum_stage,)
    raise ValueError(f"Unsupported warm-start profile: {config.warm_start_profile!r}")


def _batch_indices(
    rng: np.random.Generator, count: int, batch_size: int
) -> Iterable[NDArray[np.int64]]:
    permutation = rng.permutation(count)
    for start in range(0, count, batch_size):
        yield permutation[start : start + batch_size]


def _history_safe_force_teacher_action(stacked_observation: Observation) -> Action:
    """Return a non-oracle warm-start action from the policy's own history input."""

    frames = stacked_observation.reshape(-1, BASE_OBSERVATION_SIZE)
    latest = frames[-1]
    forces = per_finger_max_taxel_force(latest)
    low = WARM_START_TARGET_FORCE - WARM_START_FORCE_BAND
    high = WARM_START_TARGET_FORCE + WARM_START_FORCE_BAND
    stable_window = min(WARM_START_STABLE_WINDOW, len(frames))
    recent_forces = np.asarray(
        [per_finger_max_taxel_force(frame) for frame in frames[-stable_window:]],
        dtype=np.float32,
    )
    stable_contact = bool(
        np.all(
            (np.count_nonzero(recent_forces >= low, axis=1) >= 2)
            & (np.max(recent_forces, axis=1) <= high)
        )
    )
    contact_count = int(np.count_nonzero(forces >= WARM_START_CONTACT_THRESHOLD))
    lift_phase = bool(latest[43] > 0.5)
    already_lifting = bool(np.any(frames[:, 39] > 0.5))
    inferred_step = (1.0 - float(latest[44])) * POLICY_ENV_CONFIG.max_episode_steps
    late_lift = inferred_step >= WARM_START_LATE_LIFT_STEP and contact_count >= 2
    lifting = already_lifting or (lift_phase and (stable_contact or late_lift))

    if contact_count == 0:
        finger_actions = np.full(3, WARM_START_CLOSE_RATE, dtype=np.float32)
    else:
        force_floor = low * 0.85 if lifting else low
        finger_actions = np.where(
            forces < force_floor,
            WARM_START_TRIM_CLOSE_RATE,
            np.where(forces > high, -WARM_START_RELEASE_RATE, 0.0),
        ).astype(np.float32)
    palm = 1.0 if lifting else 0.0
    return np.r_[palm, np.clip(finger_actions, -1.0, 1.0)].astype(np.float32)


def _latest_frame(stacked_observation: Observation) -> Observation:
    return stacked_observation.reshape(-1, BASE_OBSERVATION_SIZE)[-1]


def _load_algorithms() -> dict[AlgorithmName, Any]:
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.utils import polyak_update
        import torch as th
        import torch.nn.functional as F
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are missing. Install stable-baselines3 and tensorboard "
            "in .venv before running blindtouch.train."
        ) from error
    try:
        from torch.func import functional_call
    except ImportError:
        from torch.nn.utils.stateless import functional_call  # type: ignore[no-redef]

    class AnchoredSAC(SAC):
        """SAC with an optional behavior-cloning anchor for warm-start demos."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.bc_anchor_observations = None
            self.bc_anchor_actions = None
            self.bc_anchor_weight = 0.0
            self.bc_anchor_batch_size = 0
            self.bc_anchor_rng = np.random.default_rng()
            self.policy_anchor_state_dict = None
            self.policy_anchor_weight = 0.0

        def set_bc_anchor(
            self,
            observations: NDArray[np.float32],
            actions: NDArray[np.float32],
            *,
            weight: float,
            batch_size: int,
            seed: int,
        ) -> None:
            self.bc_anchor_observations = th.as_tensor(
                observations, dtype=th.float32, device=self.device
            )
            self.bc_anchor_actions = th.as_tensor(actions, dtype=th.float32, device=self.device)
            self.bc_anchor_weight = float(weight)
            self.bc_anchor_batch_size = int(batch_size)
            self.bc_anchor_rng = np.random.default_rng(seed)

        def set_policy_anchor(self, *, weight: float) -> None:
            self.policy_anchor_state_dict = _snapshot_torch_state_dict(self.actor)
            self.policy_anchor_weight = float(weight)

        def _excluded_save_params(self) -> list[str]:
            return super()._excluded_save_params() + [
                "bc_anchor_observations",
                "bc_anchor_actions",
                "bc_anchor_rng",
                "policy_anchor_state_dict",
            ]

        def train(self, gradient_steps: int, batch_size: int = 64) -> None:
            self.policy.set_training_mode(True)
            optimizers = [self.actor.optimizer, self.critic.optimizer]
            if self.ent_coef_optimizer is not None:
                optimizers += [self.ent_coef_optimizer]
            self._update_learning_rate(optimizers)

            ent_coef_losses, ent_coefs = [], []
            actor_losses, critic_losses = [], []
            bc_anchor_losses, policy_anchor_losses = [], []
            for gradient_step in range(gradient_steps):
                replay_data = self.replay_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )
                discounts = (
                    replay_data.discounts if replay_data.discounts is not None else self.gamma
                )
                if self.use_sde:
                    self.actor.reset_noise()

                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    ent_coef = th.exp(self.log_ent_coef.detach())
                    assert isinstance(self.target_entropy, float)
                    ent_coef_loss = -(
                        self.log_ent_coef * (log_prob + self.target_entropy).detach()
                    ).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                else:
                    ent_coef = self.ent_coef_tensor
                ent_coefs.append(ent_coef.item())

                if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                with th.no_grad():
                    next_actions, next_log_prob = self.actor.action_log_prob(
                        replay_data.next_observations
                    )
                    next_q_values = th.cat(
                        self.critic_target(replay_data.next_observations, next_actions),
                        dim=1,
                    )
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    target_q_values = (
                        replay_data.rewards
                        + (1 - replay_data.dones) * discounts * next_q_values
                    )

                current_q_values = self.critic(replay_data.observations, replay_data.actions)
                critic_loss = 0.5 * sum(
                    F.mse_loss(current_q, target_q_values)
                    for current_q in current_q_values
                )
                critic_losses.append(critic_loss.item())

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                bc_anchor_loss = self._bc_anchor_loss()
                if bc_anchor_loss is not None:
                    actor_loss = actor_loss + self.bc_anchor_weight * bc_anchor_loss
                    bc_anchor_losses.append(float(bc_anchor_loss.detach().cpu().item()))
                policy_anchor_loss = self._policy_anchor_loss(replay_data.observations)
                if policy_anchor_loss is not None:
                    actor_loss = actor_loss + self.policy_anchor_weight * policy_anchor_loss
                    policy_anchor_losses.append(
                        float(policy_anchor_loss.detach().cpu().item())
                    )
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                if gradient_step % self.target_update_interval == 0:
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
            if ent_coef_losses:
                self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            if bc_anchor_losses:
                self.logger.record("train/bc_anchor_loss", np.mean(bc_anchor_losses))
                self.logger.record("train/bc_anchor_weight", self.bc_anchor_weight)
            if policy_anchor_losses:
                self.logger.record(
                    "train/policy_anchor_loss", np.mean(policy_anchor_losses)
                )
                self.logger.record(
                    "train/policy_anchor_weight", self.policy_anchor_weight
                )

        def _bc_anchor_loss(self) -> Any:
            if (
                self.bc_anchor_weight <= 0.0
                or self.bc_anchor_observations is None
                or self.bc_anchor_actions is None
            ):
                return None
            count = int(self.bc_anchor_observations.shape[0])
            batch_size = min(self.bc_anchor_batch_size, count)
            indices = self.bc_anchor_rng.integers(0, count, size=batch_size)
            index_tensor = th.as_tensor(indices, dtype=th.long, device=self.device)
            observations = self.bc_anchor_observations[index_tensor]
            target_actions = self.bc_anchor_actions[index_tensor]
            predicted_actions = self.actor(observations, deterministic=True)
            return F.mse_loss(predicted_actions, target_actions)

        def _policy_anchor_loss(self, observations: Any) -> Any:
            if self.policy_anchor_weight <= 0.0 or self.policy_anchor_state_dict is None:
                return None
            with th.no_grad():
                target_actions = functional_call(
                    self.actor,
                    self.policy_anchor_state_dict,
                    (observations,),
                    {"deterministic": True},
                )
            predicted_actions = self.actor(observations, deterministic=True)
            return F.mse_loss(predicted_actions, target_actions)

    return {"ppo": PPO, "sac": AnchoredSAC}


def _write_training_config(config: TrainingConfig, path: Path) -> None:
    serializable = asdict(config)
    for key in ("output_root", "checkpoint_root", "tensorboard_root"):
        serializable[key] = str(serializable[key])
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tactile BlindTouch policies with PPO or SAC.")
    parser.add_argument("--algorithm", choices=("ppo", "sac", "both"), default="both")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stage",
        choices=("robust_upright", "upright", "fragile_upright", "all_poses", "with_chassis"),
        default="upright",
    )
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument(
        "--eval-suite",
        action="append",
        choices=ALLOWED_EVALUATION_SUITES,
        dest="evaluation_suites",
        help=(
            "Suite to evaluate at every checkpoint. May be passed multiple times; "
            "defaults to validation_procedural and test_procedural_holdout."
        ),
    )
    parser.add_argument(
        "--promotion-suite",
        choices=ALLOWED_EVALUATION_SUITES,
        default=DEFAULT_PROMOTION_SUITE,
        help="Configured evaluation suite used to choose best.zip.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--tensorboard-root", type=Path, default=Path("runs/tensorboard"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-uncertified-environment",
        action="store_true",
        help="Acknowledge the failing oracle feasibility gate for an intentional dry run.",
    )
    parser.add_argument(
        "--warm-start-transitions",
        type=int,
        default=0,
        help="Behavior-clone this many scripted tactile teacher transitions before RL.",
    )
    parser.add_argument(
        "--warm-start-teacher",
        choices=("safe_force", "composed_touch"),
        default="composed_touch",
        help="Scripted touch-only teacher used for behavior-cloning warm starts.",
    )
    parser.add_argument(
        "--warm-start-profile",
        choices=("stage", "fragile_mix", "stratified_quality"),
        default="stage",
        help="Object distribution used for collecting warm-start demonstrations.",
    )
    parser.add_argument(
        "--warm-start-epochs",
        type=int,
        default=4,
        help="Supervised passes over safe-force warm-start transitions.",
    )
    parser.add_argument(
        "--warm-start-batch-size",
        type=int,
        default=256,
        help="Batch size for safe-force warm-start behavior cloning.",
    )
    parser.add_argument(
        "--warm-start-learning-rate",
        type=float,
        help=(
            "Override the behavior-cloning learning rate. Defaults to the "
            "algorithm's first-pass rate even when --learning-rate lowers RL updates."
        ),
    )
    parser.add_argument(
        "--warm-start-validation-suite",
        choices=("validation_procedural", "test_procedural_holdout", "test_pose", "test_stress"),
        default="validation_procedural",
        help="Locked suite used to gate the warm-start teacher before behavior cloning.",
    )
    parser.add_argument(
        "--warm-start-validation-limit",
        type=int,
        default=24,
        help="Number of locked cases used by the warm-start teacher gate.",
    )
    parser.add_argument(
        "--warm-start-min-safe-success-rate",
        type=float,
        default=0.10,
        help="Minimum safe-success rate required before cloning the warm-start teacher.",
    )
    parser.add_argument(
        "--warm-start-policy-gate-suite",
        choices=ALLOWED_EVALUATION_SUITES,
        default="validation_procedural",
        help="Evaluated suite used to gate the cloned policy before RL updates.",
    )
    parser.add_argument(
        "--warm-start-policy-min-safe-success-rate",
        type=float,
        default=0.10,
        help="Minimum safe-success rate required from the cloned policy before RL.",
    )
    parser.add_argument(
        "--sac-bc-anchor-weight",
        type=float,
        default=0.0,
        help="Opt-in SAC actor loss weight for preserving warm-start demonstrations.",
    )
    parser.add_argument(
        "--sac-bc-anchor-batch-size",
        type=int,
        default=256,
        help="Demonstration batch size for the SAC BC-anchor regularizer.",
    )
    parser.add_argument(
        "--sac-policy-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Opt-in SAC actor loss weight for preserving the post-BC actor on "
            "replay observations."
        ),
    )
    parser.add_argument(
        "--rl-regression-tolerance",
        type=float,
        default=0.0,
        help="Allowed safe-success drop from the post-BC checkpoint before flagging regression.",
    )
    parser.add_argument(
        "--stop-on-rl-regression",
        action="store_true",
        help="Stop a run immediately after an evaluated RL checkpoint regresses from post-BC.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override the algorithm learning rate for targeted tuning experiments.",
    )
    parser.add_argument(
        "--ppo-clip-range",
        type=float,
        help="Override PPO clip_range to make policy updates more conservative.",
    )
    parser.add_argument(
        "--ppo-target-kl",
        type=float,
        help="Set PPO target_kl to stop overly large update batches.",
    )
    parser.add_argument(
        "--sac-learning-starts",
        type=int,
        help="Override SAC learning_starts for preservation experiments.",
    )
    parser.add_argument(
        "--sac-ent-coef",
        type=float,
        help="Use a fixed SAC entropy coefficient instead of the default auto tuner.",
    )
    args = parser.parse_args()
    evaluation_suites = (
        tuple(args.evaluation_suites)
        if args.evaluation_suites is not None
        else DEFAULT_EVALUATION_SUITES
    )

    selected = ("ppo", "sac") if args.algorithm == "both" else (args.algorithm,)
    for algorithm in selected:
        result = train(
            TrainingConfig(
                algorithm=algorithm,
                total_timesteps=args.steps,
                seed=args.seed,
                curriculum_stage=args.stage,
                evaluation_frequency=args.eval_every,
                evaluation_limit=args.eval_limit,
                evaluation_suites=evaluation_suites,
                promotion_suite=args.promotion_suite,
                output_root=args.output_root,
                checkpoint_root=args.checkpoint_root,
                tensorboard_root=args.tensorboard_root,
                device=args.device,
                allow_uncertified_environment=args.allow_uncertified_environment,
                warm_start_transitions=args.warm_start_transitions,
                warm_start_teacher=args.warm_start_teacher,
                warm_start_profile=args.warm_start_profile,
                warm_start_epochs=args.warm_start_epochs,
                warm_start_batch_size=args.warm_start_batch_size,
                warm_start_learning_rate=args.warm_start_learning_rate,
                warm_start_validation_suite=args.warm_start_validation_suite,
                warm_start_validation_limit=args.warm_start_validation_limit,
                warm_start_min_safe_success_rate=args.warm_start_min_safe_success_rate,
                warm_start_policy_gate_suite=args.warm_start_policy_gate_suite,
                warm_start_policy_min_safe_success_rate=(
                    args.warm_start_policy_min_safe_success_rate
                ),
                sac_bc_anchor_weight=args.sac_bc_anchor_weight,
                sac_bc_anchor_batch_size=args.sac_bc_anchor_batch_size,
                sac_policy_anchor_weight=args.sac_policy_anchor_weight,
                rl_regression_tolerance=args.rl_regression_tolerance,
                stop_on_rl_regression=args.stop_on_rl_regression,
                learning_rate=args.learning_rate,
                ppo_clip_range=args.ppo_clip_range,
                ppo_target_kl=args.ppo_target_kl,
                sac_learning_starts=args.sac_learning_starts,
                sac_ent_coef=args.sac_ent_coef,
            )
        )
        print(
            f"{result.algorithm.upper()} trained {result.trained_timesteps} steps; "
            f"best {result.promotion_suite} safe-success={result.best_safe_success_rate:.3f}; "
            f"checkpoint={result.best_checkpoint}"
        )

if __name__ == "__main__":
    main()


__all__ = [
    "ALLOWED_EVALUATION_SUITES",
    "DEFAULT_EVALUATION_SUITES",
    "DEFAULT_PROMOTION_SUITE",
    "DEFAULT_WARM_START_LEARNING_RATES",
    "CheckpointEvaluation",
    "EvaluationSuiteName",
    "FEASIBILITY_GATE_MESSAGE",
    "POLICY_ENV_CONFIG",
    "StackedPolicyController",
    "TOUCH_TEACHER_MODES",
    "TrainingConfig",
    "TrainingResult",
    "WARM_START_GATE_MESSAGE",
    "WARM_START_POLICY_GATE_MESSAGE",
    "WarmStartTeacherController",
    "algorithm_hyperparameters",
    "build_model",
    "curriculum_sampling_config",
    "evaluate_checkpoint",
    "evaluate_policy",
    "make_training_env",
    "require_feasibility_acknowledgement",
    "train",
    "validate_warm_start_policy",
    "validate_warm_start_teacher",
    "warm_start_from_safe_force_controller",
]
