"""Locked evaluation suites and report writing for BlindTouch.

Evaluation cases carry privileged object parameters because they define the
benchmark and its report. Controllers receive only the normal environment
observation unless they are explicitly instantiated as oracle diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from .controllers import (
    Controller,
    FixedGripController,
    OracleDebugController,
    ProbeThenLiftController,
    ThresholdGripController,
)
from .env import BlindTouchEnv, EnvConfig
from .objects import (
    EpisodeObject,
    SamplingConfig,
    TRAINING_FAMILIES,
    episode_object_from_mapping,
    sample_training_object,
)


SUITE_SIZES = {
    "validation_procedural": 300,
    "test_procedural_holdout": 300,
    "test_pose": 200,
    "test_stress": 100,
}
SUITE_SEED_OFFSETS = {
    "validation_procedural": 10_000,
    "test_procedural_holdout": 50_000,
    "test_pose": 20_000,
    "test_stress": 40_000,
}
RETIRED_SUITE_ALIASES = {
    "test_household": "dev_household_seen",
    "validation_interp": "validation_procedural",
}
PROCEDURAL_SUITE_FAMILIES = tuple(TRAINING_FAMILIES)
BASELINE_ENV_CONFIG = EnvConfig(exploration_steps=0, max_episode_steps=120)
REPORT_FIELDS = (
    "controller",
    "checkpoint",
    "suite",
    "seed",
    "object_family",
    "object_name",
    "pose",
    "outcome",
    "safe_success",
    "peak_force",
    "contact_steps",
    "first_contact_step",
    "max_contacts",
    "final_grip_score",
    "slip_events",
    "slip_distance",
    "tilt",
    "exploration_steps",
    "final_lift_height",
    "episode_return",
)
FEASIBILITY_REPORT_FIELDS = (
    "seed",
    "strategy",
    "attempted_strategies",
    "object_family",
    "object_name",
    "pose",
    "shape",
    "half_size_x",
    "half_size_y",
    "half_size_z",
    "mass",
    "friction",
    "safe_force",
    "required_pad_force",
    "safety_headroom_ratio",
    "headroom_bucket",
    "outcome",
    "safe_success",
    "steps",
    "peak_force",
    "max_force_imbalance",
    "max_contacts",
    "ever_contact_finger_1",
    "ever_contact_finger_2",
    "ever_contact_finger_3",
    "final_lift_height",
    "slip_events",
    "slip_distance",
    "tilt",
)
FEASIBILITY_TRAJECTORY_PLANS: tuple[tuple[tuple[float, int], ...], ...] = (
    ((0.15, 110),),
    ((0.18, 104),),
    ((0.15, 95),),
    ((0.12, 110),),
    ((0.20, 90),),
    ((0.18, 88),),
    ((0.20, 78),),
    ((0.15, 80),),
    ((0.25, 78),),
    ((0.30, 45),),
    ((0.25, 54),),
    ((0.25, 66),),
    ((0.40, 38),),
    ((0.10, 110),),
    ((0.12, 100),),
    ((0.14, 90),),
    ((0.14, 100),),
    ((0.14, 110),),
    ((0.15, 105),),
    ((0.15, 115),),
    ((0.16, 80),),
    ((0.16, 95),),
    ((0.16, 108),),
    ((0.18, 72),),
    ((0.18, 100),),
    ((0.18, 112),),
    ((0.20, 66),),
    ((0.22, 62),),
    ((0.22, 76),),
    ((0.22, 90),),
    ((0.30, 55),),
    ((0.35, 42),),
)


@dataclass(frozen=True)
class EvaluationCase:
    """One fixed hidden object instance and its reproducibility seed."""

    suite: str
    seed: int
    object: EpisodeObject


@dataclass(frozen=True)
class EvaluationSuite:
    """A named sequence of deterministic cases used for comparisons."""

    name: str
    cases: tuple[EvaluationCase, ...]


ControllerFactory = Callable[[], Controller]
Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class ReplayResult:
    """Result metadata for a rendered deterministic replay."""

    record: dict[str, Any]
    frame_count: int
    frame_directory: Path
    video_path: Path | None


def public_controller_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only timing/outcome state to ordinary scripted or learned policies."""

    return {key: info[key] for key in ("phase", "step", "outcome")}


def _controller_info(info: Mapping[str, Any], privileged: bool) -> Mapping[str, Any]:
    return info if privileged else public_controller_info(info)


def build_locked_suite(name: str, *, limit: int | None = None) -> EvaluationSuite:
    """Build one evaluation suite from fixed seeds and auditable constructors."""

    if name not in SUITE_SIZES:
        if name in RETIRED_SUITE_ALIASES:
            replacement = RETIRED_SUITE_ALIASES[name]
            raise ValueError(
                f"Suite {name!r} is retired on main; use {replacement!r} for historical "
                "discussion or a current procedural suite for new experiments."
            )
        raise ValueError(f"Unknown evaluation suite: {name!r}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    count = SUITE_SIZES[name] if limit is None else min(limit, SUITE_SIZES[name])
    builder = {
        "validation_procedural": _validation_case,
        "test_procedural_holdout": _holdout_case,
        "test_pose": _pose_case,
        "test_stress": _stress_case,
    }[name]
    return EvaluationSuite(name=name, cases=tuple(builder(index) for index in range(count)))


def run_evaluation(
    controller_factory: ControllerFactory,
    suite: EvaluationSuite | str,
    *,
    controller_name: str,
    checkpoint: str = "",
    env_config: EnvConfig = BASELINE_ENV_CONFIG,
    privileged_controller: bool = False,
) -> list[dict[str, Any]]:
    """Run a controller on fixed cases and return one report row per episode."""

    selected_suite = build_locked_suite(suite) if isinstance(suite, str) else suite
    records: list[dict[str, Any]] = []
    env = BlindTouchEnv(config=env_config)
    try:
        for case in selected_suite.cases:
            observation, info = env.reset(seed=case.seed, options={"object": case.object})
            controller = controller_factory()
            controller.reset(observation, _controller_info(info, privileged_controller))
            episode_return = 0.0
            while True:
                observation, reward, terminated, truncated, info = env.step(
                    controller.act(observation, _controller_info(info, privileged_controller))
                )
                episode_return += reward
                if terminated or truncated:
                    break

            records.append(
                _episode_record(
                    controller_name,
                    checkpoint,
                    selected_suite.name,
                    case,
                    info,
                    episode_return,
                    env_config,
                )
            )
    finally:
        env.close()
    return records


def run_feasibility_diagnostic(
    *,
    seeds: Iterable[int] = range(200),
    sampling_config: SamplingConfig | None = None,
    env_config: EnvConfig = BASELINE_ENV_CONFIG,
    controller_factory: ControllerFactory = OracleDebugController,
) -> list[dict[str, Any]]:
    """Run privileged fixed-seed certification while recording physical failure signals."""

    records: list[dict[str, Any]] = []
    env = BlindTouchEnv(config=env_config, sampling_config=sampling_config)
    try:
        for seed in seeds:
            env.reset(seed=int(seed))
            episode_object = env.object_params
            if episode_object is None:
                raise RuntimeError("Expected a sampled episode object after reset")
            controller = controller_factory()
            records.append(
                _run_feasibility_episode(
                    env,
                    int(seed),
                    episode_object,
                    controller,
                    strategy="feedback",
                    attempted_strategies=1,
                    env_config=env_config,
                )
            )
    finally:
        env.close()
    return records


def search_feasible_trajectories(
    *,
    seeds: Iterable[int] = range(200),
    sampling_config: SamplingConfig | None = None,
    env_config: EnvConfig = BASELINE_ENV_CONFIG,
    trajectory_plans: tuple[tuple[tuple[float, int], ...], ...] = FEASIBILITY_TRAJECTORY_PLANS,
) -> list[dict[str, Any]]:
    """Certify episodes by trying a declared finite set of privileged trajectories."""

    if not trajectory_plans:
        raise ValueError("trajectory_plans must contain at least one candidate")
    records: list[dict[str, Any]] = []
    env = BlindTouchEnv(config=env_config, sampling_config=sampling_config)
    try:
        for seed in seeds:
            env.reset(seed=int(seed))
            episode_object = env.object_params
            if episode_object is None:
                raise RuntimeError("Expected a sampled episode object after reset")
            attempts: list[dict[str, Any]] = []
            for plan in trajectory_plans:
                strategy = "+".join(f"close({rate:.2f},{steps})" for rate, steps in plan)
                attempts.append(
                    _run_feasibility_episode(
                        env,
                        int(seed),
                        episode_object,
                        OracleDebugController(scripted_close_phases=plan),
                        strategy=strategy,
                        attempted_strategies=len(attempts) + 1,
                        env_config=env_config,
                    )
                )
                if attempts[-1]["safe_success"]:
                    break
            selected = min(attempts, key=_feasibility_attempt_rank).copy()
            selected["attempted_strategies"] = len(attempts)
            records.append(selected)
    finally:
        env.close()
    return records


def _run_feasibility_episode(
    env: BlindTouchEnv,
    seed: int,
    episode_object: EpisodeObject,
    controller: Controller,
    *,
    strategy: str,
    attempted_strategies: int,
    env_config: EnvConfig,
) -> dict[str, Any]:
    observation, info = env.reset(seed=seed, options={"object": episode_object})
    params = info["object_params"]
    controller.reset(observation, info)
    ever_contact = np.zeros(3, dtype=np.bool_)
    max_contacts = 0
    max_force_imbalance = 0.0
    while True:
        observation, _, terminated, truncated, info = env.step(
            controller.act(observation, info)
        )
        pad_forces = np.asarray(info["pad_forces"], dtype=np.float32)
        contacting = pad_forces > env_config.contact_force_threshold
        ever_contact |= contacting
        max_contacts = max(max_contacts, int(np.count_nonzero(contacting)))
        active_forces = pad_forces[contacting]
        if active_forces.size >= 2:
            max_force_imbalance = max(
                max_force_imbalance,
                float(np.max(active_forces) - np.min(active_forces)),
            )
        if terminated or truncated:
            break
    required_pad_force = float(params["mass"] * 9.81 / (3.0 * params["friction"]))
    headroom_ratio = float(params["safe_force"] / required_pad_force)
    return {
        "seed": seed,
        "strategy": strategy,
        "attempted_strategies": attempted_strategies,
        "object_family": params["family"],
        "object_name": params["name"],
        "pose": params["pose"],
        "shape": params["shape"],
        "half_size_x": float(params["half_size_x"]),
        "half_size_y": float(params["half_size_y"]),
        "half_size_z": float(params["half_size_z"]),
        "mass": float(params["mass"]),
        "friction": float(params["friction"]),
        "safe_force": float(params["safe_force"]),
        "required_pad_force": required_pad_force,
        "safety_headroom_ratio": headroom_ratio,
        "headroom_bucket": _headroom_bucket(headroom_ratio),
        "outcome": info["outcome"],
        "safe_success": info["outcome"] == "success",
        "steps": int(info["step"]),
        "peak_force": float(info["peak_pad_force"]),
        "max_force_imbalance": max_force_imbalance,
        "max_contacts": max_contacts,
        "ever_contact_finger_1": bool(ever_contact[0]),
        "ever_contact_finger_2": bool(ever_contact[1]),
        "ever_contact_finger_3": bool(ever_contact[2]),
        "final_lift_height": float(info["lift_height"]),
        "slip_events": int(info["slip_events"]),
        "slip_distance": float(info["cumulative_slip_distance"]),
        "tilt": float(info["tilt_radians"]),
    }


def _feasibility_attempt_rank(record: dict[str, Any]) -> tuple[int, float, float]:
    outcomes = {"success": 0, "timeout": 1, "drop": 2, "unstable": 3, "damage": 4}
    return (
        outcomes[str(record["outcome"])],
        -float(record["final_lift_height"]),
        float(record["peak_force"]),
    )


def group_feasibility_failures(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize non-success cases by geometry, pose, headroom, and outcome."""

    counts: Counter[tuple[str, str, str, str]] = Counter(
        (
            str(record["object_family"]),
            str(record["pose"]),
            str(record["headroom_bucket"]),
            str(record["outcome"]),
        )
        for record in records
        if not bool(record["safe_success"])
    )
    return [
        {
            "object_family": key[0],
            "pose": key[1],
            "headroom_bucket": key[2],
            "outcome": key[3],
            "count": count,
        }
        for key, count in sorted(counts.items())
    ]


def classify_failure_mode(record: Mapping[str, Any]) -> str:
    """Map one evaluation row to an object-agnostic failure bucket."""

    if bool(record.get("safe_success")):
        return "success"
    outcome = str(record.get("outcome") or "unknown")
    if outcome in {"damage", "drop", "unstable"}:
        return outcome
    if int(record.get("slip_events", 0) or 0) > 0:
        return "slip"
    if outcome == "timeout":
        max_contacts = int(record.get("max_contacts", 0) or 0)
        final_lift_height = float(record.get("final_lift_height", 0.0) or 0.0)
        if max_contacts < 2:
            return "timeout_no_grip"
        if final_lift_height < BASELINE_ENV_CONFIG.attempted_lift_height * 0.50:
            return "timeout_no_lift"
        if final_lift_height < BASELINE_ENV_CONFIG.lift_target_height:
            return "timeout_weak_lift"
        return "timeout_hold"
    return outcome


def summarize_evaluation_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize procedural evaluation trends without object-name tuning hooks."""

    rows = list(records)
    summary = _evaluation_subset_summary(rows)
    summary["by_family"] = {
        family: _evaluation_subset_summary(
            [record for record in rows if str(record["object_family"]) == family]
        )
        for family in sorted({str(record["object_family"]) for record in rows})
    }
    return summary


def _evaluation_subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_count = sum(bool(record["safe_success"]) for record in rows)
    outcome_counts = Counter(str(record["outcome"]) for record in rows)
    failure_counts = Counter(
        classify_failure_mode(record) for record in rows if not bool(record["safe_success"])
    )
    return {
        "episodes": len(rows),
        "safe_successes": success_count,
        "safe_success_rate": success_count / len(rows) if rows else 0.0,
        "outcomes": dict(sorted(outcome_counts.items())),
        "failure_modes": dict(sorted(failure_counts.items())),
        "mean_peak_force": _mean_record_value(rows, "peak_force"),
        "mean_slip_events": _mean_record_value(rows, "slip_events"),
        "mean_final_lift_height": _mean_record_value(rows, "final_lift_height"),
    }


def _mean_record_value(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(record[field]) for record in rows]))


def write_feasibility_report(
    records: Iterable[dict[str, Any]], output_prefix: str | Path
) -> tuple[Path, Path, Path]:
    """Write episode-level diagnostic records plus grouped failure evidence."""

    rows = list(records)
    prefix = Path(output_prefix)
    csv_path = prefix.with_suffix(".csv")
    jsonl_path = prefix.with_suffix(".jsonl")
    grouped_path = prefix.with_name(prefix.name + "_failures.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FEASIBILITY_REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_jsonl_report(rows, jsonl_path)
    grouped_path.write_text(
        json.dumps(group_feasibility_failures(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, jsonl_path, grouped_path


def _headroom_bucket(ratio: float) -> str:
    if ratio < 2.0:
        return "<2x"
    if ratio < 4.0:
        return "2-4x"
    return ">=4x"


def render_replay(
    controller_factory: ControllerFactory,
    case: EvaluationCase,
    *,
    controller_name: str,
    output_dir: str | Path,
    checkpoint: str = "",
    env_config: EnvConfig = BASELINE_ENV_CONFIG,
    privileged_controller: bool = False,
    camera_name: str = "overview",
    width: int = 1280,
    height: int = 720,
    fps: int = 25,
    encode_video: bool = True,
) -> ReplayResult:
    """Render an annotated deterministic episode without feeding pixels to control."""

    destination = Path(output_dir)
    stem = f"{case.object.name}_{controller_name}_{camera_name}"
    frame_directory = destination / f"{stem}_frames"
    frame_directory.mkdir(parents=True, exist_ok=True)
    for stale_frame in frame_directory.glob("frame_*.ppm"):
        stale_frame.unlink()
    env = BlindTouchEnv(
        config=env_config,
        render_mode="rgb_array",
        camera_name=camera_name,
        width=width,
        height=height,
    )
    frame_count = 0
    try:
        observation, info = env.reset(seed=case.seed, options={"object": case.object})
        controller = controller_factory()
        controller.reset(observation, _controller_info(info, privileged_controller))
        frame_count = _write_annotated_frame(
            env, observation, info, frame_directory, frame_count, reveal_result=False
        )
        episode_return = 0.0
        while True:
            action = controller.act(observation, _controller_info(info, privileged_controller))
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            frame_count = _write_annotated_frame(
                env,
                observation,
                info,
                frame_directory,
                frame_count,
                reveal_result=terminated or truncated,
            )
            if terminated or truncated:
                break
        record = _episode_record(
            controller_name, checkpoint, case.suite, case, info, episode_return, env_config
        )
    finally:
        env.close()

    video_path = destination / f"{stem}.mp4" if encode_video else None
    if video_path is not None:
        encode_frame_sequence(frame_directory, video_path, fps=fps)
    write_jsonl_report([record], destination / f"{stem}.jsonl")
    return ReplayResult(record, frame_count, frame_directory, video_path)


def add_overlay(
    frame: Frame, observation: NDArray[np.float32], info: Mapping[str, Any], *, reveal_result: bool
) -> Frame:
    """Draw tactile and outcome diagnostics over unused scene background."""

    image = _scene_with_claw_left(frame)
    text_scale = 2 if image.shape[0] >= 560 and image.shape[1] >= 1000 else 1
    panel_x, panel_y, panel_width, panel_height = _hud_panel_bounds(image, text_scale)
    _blend_panel(
        image,
        panel_x,
        panel_y,
        panel_width,
        panel_height,
        color=np.array([10, 17, 27], dtype=np.uint8),
        opacity=0.68,
    )

    margin = 18 if text_scale == 2 else 14
    title_y = 18 if text_scale == 2 else 12
    phase_y = 50 if text_scale == 2 else 34
    force_y = 74 if text_scale == 2 else 51
    peak_y = 98 if text_scale == 2 else 68
    label_y = 136 if text_scale == 2 else 96
    grid_top = 164 if text_scale == 2 else 114
    cell_size = 19 if text_scale == 2 else 12
    cell_pitch = 24 if text_scale == 2 else 16
    finger_spacing = 92 if text_scale == 2 else 61
    object_y = 268 if text_scale == 2 else 214
    outcome_y = 296 if text_scale == 2 else 232
    text_x = panel_x + margin

    _draw_text(
        image,
        text_x,
        panel_y + title_y,
        "BLINDTOUCH  TOUCH ONLY",
        (220, 235, 245),
        text_scale,
    )
    display_phase = "LIFT" if info["phase"] == "lift" else "PROBE"
    current_force = float(np.max(np.asarray(info["pad_forces"], dtype=np.float32)))
    _draw_text(
        image,
        text_x,
        panel_y + phase_y,
        f"PHASE: {display_phase}",
        (102, 214, 225),
        text_scale,
    )
    _draw_text(
        image,
        text_x,
        panel_y + force_y,
        f"FORCE: {current_force:.2f} N",
        (235, 220, 143),
        text_scale,
    )
    _draw_text(
        image,
        text_x,
        panel_y + peak_y,
        f"PEAK: {float(info['peak_pad_force']):.2f} N",
        (235, 220, 143),
        text_scale,
    )

    taxels = np.asarray(observation[BlindTouchEnv.OBSERVATION_LAYOUT["taxels"]]).reshape(3, 3, 3)
    for finger in range(3):
        grid_x = text_x + finger * finger_spacing
        _draw_text(
            image,
            grid_x,
            panel_y + label_y,
            f"F{finger + 1}",
            (180, 200, 214),
            text_scale,
        )
        for row in range(3):
            for column in range(3):
                strength = float(np.clip(taxels[finger, row, column], 0.0, 1.0))
                color = np.array(
                    [34 + 218 * strength, 73 + 122 * strength, 104 - 62 * strength],
                    dtype=np.uint8,
                )
                top = panel_y + grid_top + row * cell_pitch
                left = grid_x + column * cell_pitch
                image[top : top + cell_size, left : left + cell_size] = color

    object_name = str(info["object_params"]["name"]).upper() if reveal_result else "HIDDEN"
    outcome = str(info["outcome"]).upper() if reveal_result else "PENDING"
    _draw_text(
        image,
        text_x,
        panel_y + object_y,
        f"OBJECT: {object_name}",
        (220, 235, 245),
        text_scale,
    )
    outcome_color = (98, 220, 152) if outcome == "SUCCESS" else (245, 132, 105)
    _draw_text(
        image,
        text_x,
        panel_y + outcome_y,
        f"OUTCOME: {outcome}",
        outcome_color,
        text_scale,
    )
    return image


def _scene_with_claw_left(frame: Frame) -> Frame:
    """Crop the camera view so the claw occupies the left side of the full frame."""

    height, width, channels = frame.shape
    if channels != 3:
        raise ValueError("Overlay expects RGB frames")
    scene_source = _crop_scene_for_overlay(frame, width / height)
    source_height, source_width, _ = scene_source.shape
    scale = max(width / source_width, height / source_height)
    scene_width = max(1, int(source_width * scale))
    scene_height = max(1, int(source_height * scale))
    scene = _resize_nearest(scene_source, scene_height, scene_width)
    left = max(0, (scene_width - width) // 2)
    top = max(0, (scene_height - height) // 2)
    return scene[top : top + height, left : left + width].copy()


def _crop_scene_for_overlay(frame: Frame, target_aspect: float) -> Frame:
    """Crop the robot camera feed for a full-frame composition with right-side HUD room."""

    source_height, source_width, _ = frame.shape
    crop_height = max(1, int(round(source_height * 0.72)))
    crop_width = max(1, int(round(crop_height * target_aspect)))
    if crop_width > source_width:
        crop_width = source_width
        crop_height = max(1, int(round(crop_width / target_aspect)))
    if crop_height > source_height:
        crop_height = source_height
        crop_width = max(1, int(round(crop_height * target_aspect)))

    center_x = int(round(source_width * 0.64))
    left = int(np.clip(center_x - crop_width // 2, 0, source_width - crop_width))
    top = 0
    return frame[top : top + crop_height, left : left + crop_width]


def _hud_panel_bounds(image: Frame, text_scale: int) -> tuple[int, int, int, int]:
    height, width, _ = image.shape
    if text_scale == 2:
        panel_width = min(370, max(344, int(width * 0.29)))
        panel_height = min(328, height - 40)
        x = width - panel_width - 24
        y = 24
    else:
        panel_width = min(232, max(218, int(width * 0.35)))
        panel_height = min(252, height - 28)
        x = width - panel_width - 14
        y = 14
    return max(0, x), max(0, y), panel_width, panel_height


def _blend_panel(
    image: Frame,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    color: NDArray[np.uint8],
    opacity: float,
) -> None:
    bottom = min(image.shape[0], y + height)
    right = min(image.shape[1], x + width)
    region = image[y:bottom, x:right]
    region[:] = ((1.0 - opacity) * region + opacity * color).astype(np.uint8)
    border = np.array([68, 83, 103], dtype=np.uint8)
    image[y:bottom, x : min(x + 2, right)] = border
    image[y:bottom, max(x, right - 2) : right] = border
    image[y : min(y + 2, bottom), x:right] = border
    image[max(y, bottom - 2) : bottom, x:right] = border


def _resize_nearest(frame: Frame, height: int, width: int) -> Frame:
    source_height, source_width, _ = frame.shape
    y_indices = np.minimum(
        (np.arange(height, dtype=np.float64) * source_height / height).astype(np.int64),
        source_height - 1,
    )
    x_indices = np.minimum(
        (np.arange(width, dtype=np.float64) * source_width / width).astype(np.int64),
        source_width - 1,
    )
    return frame[y_indices[:, None], x_indices]


def encode_frame_sequence(frame_directory: str | Path, path: str | Path, *, fps: int = 25) -> Path:
    """Encode PPM frames to an MP4 with the system ffmpeg executable."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for MP4 export but was not found")
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    encoder = next(
        (candidate for candidate in ("libx264", "libopenh264", "mpeg4") if candidate in encoders),
        None,
    )
    if encoder is None:
        raise RuntimeError("ffmpeg does not expose a supported MP4 video encoder")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(Path(frame_directory) / "frame_%05d.ppm"),
            "-c:v",
            encoder,
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        check=True,
    )
    return destination


def _episode_record(
    controller_name: str,
    checkpoint: str,
    suite_name: str,
    case: EvaluationCase,
    info: Mapping[str, Any],
    episode_return: float,
    env_config: EnvConfig,
) -> dict[str, Any]:
    params = info["object_params"]
    return {
        "controller": controller_name,
        "checkpoint": checkpoint,
        "suite": suite_name,
        "seed": case.seed,
        "object_family": params["family"],
        "object_name": params["name"],
        "pose": params["pose"],
        "outcome": info["outcome"],
        "safe_success": info["outcome"] == "success",
        "peak_force": float(info["peak_pad_force"]),
        "contact_steps": int(info["contact_steps"]),
        "first_contact_step": info["first_contact_step"],
        "max_contacts": int(info["max_contact_count"]),
        "final_grip_score": float(info["grip_score"]),
        "slip_events": int(info["slip_events"]),
        "slip_distance": float(info["cumulative_slip_distance"]),
        "tilt": float(info["tilt_radians"]),
        "exploration_steps": env_config.exploration_steps,
        "final_lift_height": float(info["lift_height"]),
        "episode_return": float(episode_return),
    }


def _write_annotated_frame(
    env: BlindTouchEnv,
    observation: NDArray[np.float32],
    info: Mapping[str, Any],
    frame_directory: Path,
    frame_count: int,
    *,
    reveal_result: bool,
) -> int:
    frame = env.render()
    if frame is None:
        raise RuntimeError("Frame capture requires an rgb_array environment")
    annotated = add_overlay(frame, observation, info, reveal_result=reveal_result)
    _write_ppm(frame_directory / f"frame_{frame_count:05d}.ppm", annotated)
    return frame_count + 1


def _write_ppm(path: Path, frame: Frame) -> None:
    height, width, channels = frame.shape
    if channels != 3:
        raise ValueError("PPM output requires RGB frames")
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + frame.tobytes())


FONT_5X7: dict[str, tuple[str, ...]] = {
    " ": ("00000",) * 7,
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def _draw_text(
    image: Frame,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    scale: int,
) -> None:
    cursor = x
    for character in text.upper():
        glyph = FONT_5X7.get(character, FONT_5X7["?"])
        for row, pixels in enumerate(glyph):
            for column, pixel in enumerate(pixels):
                if pixel == "1":
                    top = y + row * scale
                    left = cursor + column * scale
                    image[top : top + scale, left : left + scale] = color
        cursor += 6 * scale


def write_csv_report(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write report rows in a stable column order."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return destination


def write_jsonl_report(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write machine-readable report rows without non-deterministic metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return destination


def _validation_case(index: int) -> EvaluationCase:
    return _procedural_case("validation_procedural", index)


def _holdout_case(index: int) -> EvaluationCase:
    return _procedural_case("test_procedural_holdout", index)


def _procedural_case(suite_name: str, index: int) -> EvaluationCase:
    seed = SUITE_SEED_OFFSETS[suite_name] + index
    family = PROCEDURAL_SUITE_FAMILIES[index % len(PROCEDURAL_SUITE_FAMILIES)]
    episode_object = sample_training_object(
        np.random.default_rng(seed),
        SamplingConfig(training_families=(family,)),
    )
    episode_object = replace(
        episode_object,
        evaluation_tags=episode_object.evaluation_tags + (suite_name,),
    )
    return EvaluationCase(suite_name, seed, episode_object)


def _pose_case(index: int) -> EvaluationCase:
    seed = SUITE_SEED_OFFSETS["test_pose"] + index
    family = "container" if index % 2 == 0 else "package"
    sampled = sample_training_object(
        np.random.default_rng(seed), SamplingConfig(training_families=(family,))
    )
    pose = "side_x" if family == "container" or index % 4 == 1 else "side_y"
    return EvaluationCase("test_pose", seed, _replace_pose(sampled, pose))


def _stress_case(index: int) -> EvaluationCase:
    seed = SUITE_SEED_OFFSETS["test_stress"] + index
    rng = np.random.default_rng(seed)
    family = "container" if index % 2 == 0 else "package"
    sampled = sample_training_object(rng, SamplingConfig(training_families=(family,)))
    pose = "side_x" if family == "container" or index % 4 == 1 else "side_y"
    posed = _replace_pose(sampled, pose)
    friction = float(rng.uniform(0.20, 0.30) if family == "container" else rng.uniform(0.30, 0.40))
    mass = min(posed.mass * float(rng.uniform(1.05, 1.20)), 0.200)
    required_pad_force = mass * 9.81 / (3.0 * friction)
    episode_object = replace(
        posed,
        mass=mass,
        friction=friction,
        safe_force=max(0.40, required_pad_force * 1.12),
        evaluation_tags=posed.evaluation_tags + ("stress", "awkward_pose"),
    )
    return EvaluationCase("test_stress", seed, episode_object)


def _replace_pose(episode_object: EpisodeObject, pose: str) -> EpisodeObject:
    return episode_object_from_mapping(
        {
            "family": episode_object.family,
            "name": episode_object.name,
            "shape": episode_object.shape,
            "pose": pose,
            "half_size_x": episode_object.half_size_x,
            "half_size_y": episode_object.half_size_y,
            "half_size_z": episode_object.half_size_z,
            "mass": episode_object.mass,
            "friction": episode_object.friction,
            "safe_force": episode_object.safe_force,
            "x_offset": episode_object.x_offset,
            "y_offset": episode_object.y_offset,
            "yaw": episode_object.yaw,
            "visual_style": episode_object.visual_style,
            "evaluation_tags": episode_object.evaluation_tags,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic BlindTouch evaluation.")
    parser.add_argument("--suite", choices=tuple(SUITE_SIZES), default="validation_procedural")
    parser.add_argument(
        "--feasibility",
        action="store_true",
        help="Write the privileged fixed-seed feasibility report instead of a suite report.",
    )
    parser.add_argument(
        "--trajectory-search",
        action="store_true",
        help="Use the declared finite privileged trajectory set with --feasibility.",
    )
    parser.add_argument("--camera", choices=("overview", "closeup"), default="overview")
    parser.add_argument(
        "--controller",
        choices=("fixed", "threshold", "probe", "oracle"),
        default="probe",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--frames-only", action="store_true", help="Skip MP4 encoding.")
    args = parser.parse_args()

    factories: dict[str, ControllerFactory] = {
        "fixed": FixedGripController,
        "threshold": ThresholdGripController,
        "probe": ProbeThenLiftController,
        "oracle": OracleDebugController,
    }
    privileged_controller = args.controller == "oracle"
    if args.feasibility:
        count = args.limit or 200
        if args.trajectory_search:
            records = search_feasible_trajectories(seeds=range(count))
            prefix = args.output_dir / "feasibility_trajectory_search"
        else:
            records = run_feasibility_diagnostic(seeds=range(count))
            prefix = args.output_dir / "feasibility_oracle"
        csv_path, jsonl_path, grouped_path = write_feasibility_report(records, prefix)
        outcomes = Counter(record["outcome"] for record in records)
        print(
            f"Wrote {len(records)} feasibility cases to {csv_path}, {jsonl_path}, "
            f"and {grouped_path}: {dict(outcomes)}"
        )
        return
    suite = build_locked_suite(args.suite, limit=args.limit)
    records = run_evaluation(
        factories[args.controller],
        suite,
        controller_name=args.controller,
        checkpoint=args.checkpoint,
        privileged_controller=privileged_controller,
    )
    prefix = args.output_dir / f"{args.suite}_{args.controller}"
    csv_path = write_csv_report(records, prefix.with_suffix(".csv"))
    jsonl_path = write_jsonl_report(records, prefix.with_suffix(".jsonl"))
    outcomes = Counter(record["outcome"] for record in records)
    print(f"Wrote {len(records)} episodes to {csv_path} and {jsonl_path}: {dict(outcomes)}")


if __name__ == "__main__":
    main()


__all__ = [
    "BASELINE_ENV_CONFIG",
    "EvaluationCase",
    "EvaluationSuite",
    "FEASIBILITY_REPORT_FIELDS",
    "FEASIBILITY_TRAJECTORY_PLANS",
    "REPORT_FIELDS",
    "ReplayResult",
    "RETIRED_SUITE_ALIASES",
    "SUITE_SIZES",
    "add_overlay",
    "build_locked_suite",
    "classify_failure_mode",
    "encode_frame_sequence",
    "group_feasibility_failures",
    "public_controller_info",
    "render_replay",
    "run_evaluation",
    "run_feasibility_diagnostic",
    "search_feasible_trajectories",
    "summarize_evaluation_records",
    "write_csv_report",
    "write_feasibility_report",
    "write_jsonl_report",
]
