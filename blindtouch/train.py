"""PPO and SAC training entry point for BlindTouch.

This module keeps Stable-Baselines3 as an optional runtime dependency so the
simulation, rendering, and tests remain usable before training packages are
installed. Production learning is deliberately gated until the randomized
oracle feasibility diagnostic meets the readiness target recorded in todo.txt.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from .controllers import per_finger_max_taxel_force
from .env import BlindTouchEnv, EnvConfig, ObservationHistory
from .evaluate import (
    HOUSEHOLD_NAMES,
    EvaluationSuite,
    ReplayResult,
    build_locked_suite,
    demo_case,
    render_replay,
    run_evaluation,
    write_csv_report,
    write_jsonl_report,
)
from .objects import SamplingConfig


AlgorithmName = Literal["ppo", "sac"]
CurriculumStage = Literal["upright", "all_poses", "with_chassis"]
Observation = NDArray[np.float32]
Action = NDArray[np.float32]
POLICY_ENV_CONFIG = EnvConfig(max_episode_steps=140)
BASE_OBSERVATION_SIZE = 45
WARM_START_TARGET_FORCE = 0.40
WARM_START_FORCE_BAND = 0.12
WARM_START_CONTACT_THRESHOLD = 0.035
WARM_START_CLOSE_RATE = 0.18
WARM_START_TRIM_CLOSE_RATE = 0.08
WARM_START_RELEASE_RATE = 0.08
WARM_START_STABLE_WINDOW = 8
WARM_START_LATE_LIFT_STEP = 120
FEASIBILITY_GATE_MESSAGE = (
    "Only the upright stage-1 curriculum is currently certified for training. "
    "Use --allow-uncertified-environment only for deliberate pipeline dry runs "
    "on all_poses or with_chassis."
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
    output_root: Path = Path("runs")
    checkpoint_root: Path = Path("checkpoints")
    tensorboard_root: Path = Path("runs/tensorboard")
    device: str = "auto"
    allow_uncertified_environment: bool = False
    warm_start_transitions: int = 0
    warm_start_epochs: int = 4
    warm_start_batch_size: int = 256

    def __post_init__(self) -> None:
        if self.algorithm not in {"ppo", "sac"}:
            raise ValueError(f"Unsupported algorithm: {self.algorithm!r}")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if self.history_length < 1:
            raise ValueError("history_length must be positive")
        if self.curriculum_stage not in {"upright", "all_poses", "with_chassis"}:
            raise ValueError(f"Unsupported curriculum stage: {self.curriculum_stage!r}")
        if self.evaluation_frequency < 1:
            raise ValueError("evaluation_frequency must be positive")
        if self.evaluation_limit is not None and self.evaluation_limit < 1:
            raise ValueError("evaluation_limit must be positive when provided")
        if self.warm_start_transitions < 0:
            raise ValueError("warm_start_transitions cannot be negative")
        if self.warm_start_epochs < 1:
            raise ValueError("warm_start_epochs must be positive")
        if self.warm_start_batch_size < 1:
            raise ValueError("warm_start_batch_size must be positive")

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
    best_safe_success_rate: float


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


def algorithm_hyperparameters(algorithm: AlgorithmName) -> dict[str, Any]:
    """Return the fixed first-pass model settings from the project roadmap."""

    common: dict[str, Any] = {
        "learning_rate": 3e-4,
        "batch_size": 256,
        "gamma": 0.99,
        "policy_kwargs": {"net_arch": [256, 256]},
    }
    if algorithm == "ppo":
        return {
            **common,
            "n_steps": 2048,
            "gae_lambda": 0.95,
        }
    if algorithm == "sac":
        return {
            **common,
            "buffer_size": 1_000_000,
            "learning_starts": 10_000,
            "ent_coef": "auto",
        }
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

    if stage == "upright":
        return SamplingConfig(
            training_families=("rounded", "container", "package"),
            allowed_poses=("upright",),
            offset_range=(-0.002, 0.002),
            safe_force_margin=3.0,
        )
    if stage == "all_poses":
        return SamplingConfig(training_families=("rounded", "container", "package"))
    if stage == "with_chassis":
        return SamplingConfig()
    raise ValueError(f"Unsupported curriculum stage: {stage!r}")


def require_feasibility_acknowledgement(
    curriculum_stage: CurriculumStage, allow_uncertified_environment: bool
) -> None:
    """Refuse accidental learning on curriculum stages not yet certified."""

    if curriculum_stage != "upright" and not allow_uncertified_environment:
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


def render_learned_policy_replay(
    model: PredictivePolicy,
    *,
    algorithm: AlgorithmName,
    checkpoint: str | Path,
    object_name: str = "orange",
    pose: str | None = None,
    output_dir: str | Path = Path("renders/policies"),
    history_length: int = ObservationHistory.DEFAULT_HISTORY_LENGTH,
    deterministic: bool = True,
    env_config: EnvConfig = POLICY_ENV_CONFIG,
    camera_name: str = "overview",
    width: int = 1280,
    height: int = 720,
    fps: int = 25,
    encode_video: bool = True,
) -> ReplayResult:
    """Render one household-object replay for a loaded stacked-observation policy."""

    return render_replay(
        lambda: StackedPolicyController(
            model, history_length=history_length, deterministic=deterministic
        ),
        demo_case(object_name, pose),
        controller_name=algorithm,
        checkpoint=str(checkpoint),
        output_dir=output_dir,
        env_config=env_config,
        camera_name=camera_name,
        width=width,
        height=height,
        fps=fps,
        encode_video=encode_video,
    )


def render_policy_checkpoint(
    *,
    algorithm: AlgorithmName,
    checkpoint: str | Path,
    object_name: str = "orange",
    pose: str | None = None,
    output_dir: str | Path = Path("renders/policies"),
    history_length: int = ObservationHistory.DEFAULT_HISTORY_LENGTH,
    device: str = "auto",
    deterministic: bool = True,
    camera_name: str = "overview",
    width: int = 1280,
    height: int = 720,
    fps: int = 25,
    encode_video: bool = True,
) -> ReplayResult:
    """Load an SB3 checkpoint and render one deterministic learned-policy replay."""

    model_class = _load_algorithms()[algorithm]
    loaded_model = model_class.load(str(checkpoint), device=device)
    return render_learned_policy_replay(
        loaded_model,
        algorithm=algorithm,
        checkpoint=checkpoint,
        object_name=object_name,
        pose=pose,
        output_dir=output_dir,
        history_length=history_length,
        deterministic=deterministic,
        camera_name=camera_name,
        width=width,
        height=height,
        fps=fps,
        encode_video=encode_video,
    )


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
        **algorithm_hyperparameters(config.algorithm),
    )


def train(config: TrainingConfig) -> TrainingResult:
    """Train one algorithm, checkpoint it, and evaluate fixed validation cases."""

    require_feasibility_acknowledgement(
        config.curriculum_stage, config.allow_uncertified_environment
    )
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
            f"{config.algorithm.upper()} safe-force warm start: "
            f"{config.warm_start_transitions} transitions, final_loss={loss:.4f}"
        )
    best_score: tuple[float, float] | None = None
    best_checkpoint = checkpoint_directory / "best.zip"
    latest_report = report_directory / "validation_interp_step_0.csv"
    next_evaluation = min(config.evaluation_frequency, config.total_timesteps)
    try:
        if config.warm_start_transitions > 0:
            checkpoint_base = checkpoint_directory / "step_0_warm_start"
            model.save(str(checkpoint_base))
            checkpoint = checkpoint_base.with_suffix(".zip")
            loaded_model = model_class.load(str(checkpoint), device=config.device)
            suite = build_locked_suite("validation_interp", limit=config.evaluation_limit)
            prefix = report_directory / "validation_interp_step_0_warm_start"
            records = evaluate_policy(
                loaded_model,
                suite,
                algorithm=config.algorithm,
                checkpoint=str(checkpoint),
                history_length=config.history_length,
                output_prefix=prefix,
            )
            latest_report = prefix.with_suffix(".csv")
            best_score = _policy_score(records)
            shutil.copyfile(checkpoint, best_checkpoint)
        while model.num_timesteps < config.total_timesteps:
            remaining = next_evaluation - model.num_timesteps
            model.learn(total_timesteps=max(1, remaining), reset_num_timesteps=False)
            step = int(model.num_timesteps)
            checkpoint_base = checkpoint_directory / f"step_{step}"
            model.save(str(checkpoint_base))
            checkpoint = checkpoint_base.with_suffix(".zip")
            loaded_model = model_class.load(str(checkpoint), device=config.device)
            suite = build_locked_suite("validation_interp", limit=config.evaluation_limit)
            prefix = report_directory / f"validation_interp_step_{step}"
            records = evaluate_policy(
                loaded_model,
                suite,
                algorithm=config.algorithm,
                checkpoint=str(checkpoint),
                history_length=config.history_length,
                output_prefix=prefix,
            )
            latest_report = prefix.with_suffix(".csv")
            score = _policy_score(records)
            if best_score is None or score > best_score:
                shutil.copyfile(checkpoint, best_checkpoint)
                best_score = score
            if model.num_timesteps >= config.total_timesteps:
                break
            next_evaluation = min(
                config.total_timesteps, next_evaluation + config.evaluation_frequency
            )
    finally:
        env.close()

    if best_score is None:
        raise RuntimeError("Training completed without evaluating a checkpoint")
    final_checkpoint = checkpoint_directory / f"step_{int(model.num_timesteps)}.zip"
    return TrainingResult(
        algorithm=config.algorithm,
        trained_timesteps=int(model.num_timesteps),
        final_checkpoint=final_checkpoint,
        best_checkpoint=best_checkpoint,
        latest_report=latest_report,
        best_safe_success_rate=best_score[0],
    )


def _policy_score(records: list[dict[str, Any]]) -> tuple[float, float]:
    success_rate = sum(bool(record["safe_success"]) for record in records) / len(records)
    mean_peak_force = float(np.mean([record["peak_force"] for record in records]))
    return success_rate, -mean_peak_force


def warm_start_from_safe_force_controller(model: Any, config: TrainingConfig) -> float:
    """Behavior-clone a small tactile force-regulation prior into the actor."""

    observations, actions = _collect_safe_force_demonstrations(config)
    try:
        import torch as th
    except ImportError as error:
        raise RuntimeError("Scripted warm start requires PyTorch from Stable-Baselines3") from error

    obs_tensor = th.as_tensor(observations, dtype=th.float32, device=model.device)
    action_tensor = th.as_tensor(actions, dtype=th.float32, device=model.device)
    rng = np.random.default_rng(config.seed + 29_000)
    final_loss = 0.0
    for _ in range(config.warm_start_epochs):
        for indices in _batch_indices(
            rng, len(observations), config.warm_start_batch_size
        ):
            batch_obs = obs_tensor[indices]
            batch_actions = action_tensor[indices]
            if config.algorithm == "ppo":
                optimizer = model.policy.optimizer
                predicted_actions = model.policy.get_distribution(batch_obs).mode()
            else:
                optimizer = model.actor.optimizer
                predicted_actions = model.actor(batch_obs, deterministic=True)
            loss = (predicted_actions - batch_actions).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
    return final_loss


def _collect_safe_force_demonstrations(
    config: TrainingConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    observations: list[Observation] = []
    actions: list[Action] = []
    env = make_training_env(
        history_length=config.history_length,
        curriculum_stage=config.curriculum_stage,
    )
    seed = config.seed + 17_000
    try:
        stacked_observation, _ = env.reset(seed=seed)
        while len(observations) < config.warm_start_transitions:
            action = _history_safe_force_teacher_action(stacked_observation)
            observations.append(stacked_observation.copy())
            actions.append(action.copy())
            stacked_observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                seed += 1
                stacked_observation, _ = env.reset(seed=seed)
    finally:
        env.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


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


def _load_algorithms() -> dict[AlgorithmName, Any]:
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are missing. Install stable-baselines3 and tensorboard "
            "in .venv before running blindtouch.train."
        ) from error
    return {"ppo": PPO, "sac": SAC}


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
        choices=("upright", "all_poses", "with_chassis"),
        default="upright",
    )
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--tensorboard-root", type=Path, default=Path("runs/tensorboard"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--render-demo",
        choices=HOUSEHOLD_NAMES,
        help="After training each selected algorithm, render its best checkpoint on this demo object.",
    )
    parser.add_argument("--render-pose", help="Named pose for --render-demo, when supported.")
    parser.add_argument("--render-output-dir", type=Path, default=Path("renders/policies"))
    parser.add_argument("--camera", choices=("overview", "closeup"), default="overview")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--frames-only", action="store_true", help="Skip MP4 encoding for replays.")
    parser.add_argument(
        "--render-checkpoint",
        type=Path,
        help="Load and render an existing checkpoint instead of training.",
    )
    parser.add_argument(
        "--allow-uncertified-environment",
        action="store_true",
        help="Acknowledge the failing oracle feasibility gate for an intentional dry run.",
    )
    parser.add_argument(
        "--warm-start-transitions",
        type=int,
        default=0,
        help="Behavior-clone this many tactile safe-force transitions before RL.",
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
    args = parser.parse_args()

    selected = ("ppo", "sac") if args.algorithm == "both" else (args.algorithm,)
    if args.render_checkpoint is not None:
        if args.algorithm == "both":
            parser.error("--render-checkpoint requires --algorithm ppo or --algorithm sac")
        replay = render_policy_checkpoint(
            algorithm=args.algorithm,
            checkpoint=args.render_checkpoint,
            object_name=args.render_demo or "orange",
            pose=args.render_pose,
            output_dir=args.render_output_dir,
            device=args.device,
            camera_name=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            encode_video=not args.frames_only,
        )
        artifact = replay.video_path or replay.frame_directory
        print(
            f"Rendered {args.algorithm.upper()} replay with {replay.frame_count} frames to "
            f"{artifact}: {replay.record['outcome']} "
            f"peak_force={replay.record['peak_force']:.3f}"
        )
        return

    for algorithm in selected:
        result = train(
            TrainingConfig(
                algorithm=algorithm,
                total_timesteps=args.steps,
                seed=args.seed,
                curriculum_stage=args.stage,
                evaluation_frequency=args.eval_every,
                evaluation_limit=args.eval_limit,
                output_root=args.output_root,
                checkpoint_root=args.checkpoint_root,
                tensorboard_root=args.tensorboard_root,
                device=args.device,
                allow_uncertified_environment=args.allow_uncertified_environment,
                warm_start_transitions=args.warm_start_transitions,
                warm_start_epochs=args.warm_start_epochs,
                warm_start_batch_size=args.warm_start_batch_size,
            )
        )
        print(
            f"{result.algorithm.upper()} trained {result.trained_timesteps} steps; "
            f"best validation safe-success={result.best_safe_success_rate:.3f}; "
            f"checkpoint={result.best_checkpoint}"
        )
        if args.render_demo:
            replay = render_policy_checkpoint(
                algorithm=algorithm,
                checkpoint=result.best_checkpoint,
                object_name=args.render_demo,
                pose=args.render_pose,
                output_dir=args.render_output_dir,
                device=args.device,
                camera_name=args.camera,
                width=args.width,
                height=args.height,
                fps=args.fps,
                encode_video=not args.frames_only,
            )
            artifact = replay.video_path or replay.frame_directory
            print(
                f"Rendered {algorithm.upper()} replay with {replay.frame_count} frames to "
                f"{artifact}: {replay.record['outcome']} "
                f"peak_force={replay.record['peak_force']:.3f}"
            )


if __name__ == "__main__":
    main()


__all__ = [
    "FEASIBILITY_GATE_MESSAGE",
    "POLICY_ENV_CONFIG",
    "StackedPolicyController",
    "TrainingConfig",
    "TrainingResult",
    "algorithm_hyperparameters",
    "build_model",
    "curriculum_sampling_config",
    "evaluate_policy",
    "make_training_env",
    "render_learned_policy_replay",
    "render_policy_checkpoint",
    "require_feasibility_acknowledgement",
    "train",
    "warm_start_from_safe_force_controller",
]
