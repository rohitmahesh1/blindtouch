import csv
import json
from typing import Any, Mapping

import numpy as np

from blindtouch.controllers import FixedGripController
from blindtouch.env import BlindTouchEnv, EnvConfig, ObservationHistory
from blindtouch.evaluate import (
    FEASIBILITY_REPORT_FIELDS,
    REPORT_FIELDS,
    SUITE_SIZES,
    add_overlay,
    build_locked_suite,
    demo_case,
    group_feasibility_failures,
    public_controller_info,
    run_feasibility_diagnostic,
    run_evaluation,
    search_feasible_trajectories,
    write_csv_report,
    write_feasibility_report,
    write_jsonl_report,
)


FIXED_OBJECT = {
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
}


def test_observation_history_stacks_eight_frames_and_clears_on_reset() -> None:
    env = ObservationHistory(BlindTouchEnv(config=EnvConfig(exploration_steps=0)))
    observation, _ = env.reset(options={"object_params": FIXED_OBJECT})

    assert observation.shape == (360,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    frames = observation.reshape(8, 45)
    np.testing.assert_allclose(frames, np.repeat(frames[-1][None, :], 8, axis=0))

    action = np.array([0.0, 0.2, 0.2, 0.2], dtype=np.float32)
    next_observation, _, _, _, _ = env.step(action)
    next_frames = next_observation.reshape(8, 45)
    np.testing.assert_allclose(next_frames[:-1], frames[1:])
    np.testing.assert_allclose(next_frames[-1, 39:43], action)

    reset_observation, _ = env.reset(options={"object_params": FIXED_OBJECT})
    reset_frames = reset_observation.reshape(8, 45)
    np.testing.assert_allclose(
        reset_frames, np.repeat(reset_frames[-1][None, :], 8, axis=0)
    )
    np.testing.assert_allclose(reset_frames[-1, 39:43], 0.0)
    env.close()


def test_stacked_policy_observation_does_not_reveal_safe_force() -> None:
    env = ObservationHistory(BlindTouchEnv())
    fragile, _ = env.reset(options={"object_params": {**FIXED_OBJECT, "safe_force": 0.1}})
    robust, _ = env.reset(options={"object_params": {**FIXED_OBJECT, "safe_force": 10.0}})
    np.testing.assert_allclose(fragile, robust)
    env.close()


def test_locked_suites_have_fixed_sizes_and_reproducible_metadata() -> None:
    for name, count in SUITE_SIZES.items():
        first = build_locked_suite(name)
        second = build_locked_suite(name)
        assert len(first.cases) == count
        assert first == second

    pose_cases = build_locked_suite("test_pose").cases
    assert all(
        case.object.pose in {"side_x", "side_y"} and case.object.family in {"container", "package"}
        for case in pose_cases
    )
    household_cases = build_locked_suite("test_household").cases
    assert {case.object.name for case in household_cases} == {
        "orange",
        "toy_car",
        "soap_bar",
        "tomato",
    }
    assert all("perturbed" in case.object.evaluation_tags for case in household_cases)
    assert all(
        "stress" in case.object.evaluation_tags
        for case in build_locked_suite("test_stress").cases
    )


def test_scripted_evaluation_reports_are_repeatable_and_complete(tmp_path) -> None:
    suite = build_locked_suite("test_household", limit=4)
    first = run_evaluation(FixedGripController, suite, controller_name="fixed")
    second = run_evaluation(FixedGripController, suite, controller_name="fixed")
    assert first == second

    first_csv = write_csv_report(first, tmp_path / "first.csv")
    second_csv = write_csv_report(second, tmp_path / "second.csv")
    first_jsonl = write_jsonl_report(first, tmp_path / "first.jsonl")
    second_jsonl = write_jsonl_report(second, tmp_path / "second.jsonl")
    assert first_csv.read_text(encoding="utf-8") == second_csv.read_text(encoding="utf-8")
    assert first_jsonl.read_text(encoding="utf-8") == second_jsonl.read_text(encoding="utf-8")

    with first_csv.open(newline="", encoding="utf-8") as report:
        rows = list(csv.DictReader(report))
    assert tuple(rows[0]) == REPORT_FIELDS
    assert len(rows) == 4
    assert all(row["suite"] == "test_household" for row in rows)
    assert all("object_params" not in row for row in rows)

    json_rows = [
        json.loads(line) for line in first_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert json_rows == first


class PublicOnlyController:
    def reset(self, observation: np.ndarray, info: Mapping[str, Any]) -> None:
        assert observation.shape == (45,)
        assert set(info) == {"phase", "step", "outcome"}

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        assert observation.shape == (45,)
        assert set(info) == {"phase", "step", "outcome"}
        return np.zeros(4, dtype=np.float32)


def test_evaluator_does_not_supply_hidden_object_or_render_state_to_controller() -> None:
    public = public_controller_info(
        {"phase": "probe", "step": 1, "outcome": None, "object_params": {"name": "orange"}}
    )
    assert public == {"phase": "probe", "step": 1, "outcome": None}

    records = run_evaluation(
        PublicOnlyController,
        build_locked_suite("test_household", limit=1),
        controller_name="public_only",
        env_config=EnvConfig(exploration_steps=0, max_episode_steps=2),
    )
    assert records[0]["object_name"] == "orange"
    assert records[0]["outcome"] == "timeout"


def test_overlay_draws_touch_panel_but_hides_object_identity_until_completion() -> None:
    frame = np.zeros((360, 520, 3), dtype=np.uint8)
    observation = np.zeros(45, dtype=np.float32)
    observation[12:21] = 0.75
    info = {
        "phase": "explore",
        "pad_forces": np.array([0.2, 0.0, 0.0], dtype=np.float32),
        "peak_pad_force": 0.2,
        "object_params": {"name": "orange"},
        "outcome": None,
    }
    hidden = add_overlay(frame, observation, info, reveal_result=False)
    revealed = add_overlay(
        frame,
        observation,
        {**info, "phase": "lift", "outcome": "damage"},
        reveal_result=True,
    )

    assert hidden.shape == frame.shape
    assert np.any(hidden != frame)
    assert np.any(hidden != revealed)
    assert demo_case("orange").object.name == "orange"


def test_feasibility_report_preserves_seeded_physics_and_grouped_failures(tmp_path) -> None:
    records = run_feasibility_diagnostic(
        seeds=range(3), env_config=EnvConfig(exploration_steps=0, max_episode_steps=2)
    )
    assert len(records) == 3
    assert set(records[0]) == set(FEASIBILITY_REPORT_FIELDS)
    assert all(record["outcome"] == "timeout" for record in records)
    assert all(record["required_pad_force"] > 0.0 for record in records)
    assert all(record["headroom_bucket"] in {"<2x", "2-4x", ">=4x"} for record in records)

    grouped = group_feasibility_failures(records)
    assert sum(group["count"] for group in grouped) == 3
    csv_path, jsonl_path, grouped_path = write_feasibility_report(
        records, tmp_path / "oracle_gate"
    )
    assert csv_path.exists()
    assert jsonl_path.exists()
    assert json.loads(grouped_path.read_text(encoding="utf-8")) == grouped


def test_feasibility_trajectory_search_records_declared_oracle_attempts() -> None:
    records = search_feasible_trajectories(
        seeds=range(1),
        env_config=EnvConfig(exploration_steps=0, max_episode_steps=2),
        trajectory_plans=(((0.15, 1),), ((0.20, 1),)),
    )
    assert len(records) == 1
    assert records[0]["attempted_strategies"] == 2
    assert records[0]["strategy"] in {"close(0.15,1)", "close(0.20,1)"}
