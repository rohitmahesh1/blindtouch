from collections import Counter
from typing import Any

import numpy as np

from blindtouch.controllers import (
    FixedGripController,
    OracleDebugController,
    ProbeThenLiftController,
    ThresholdGripController,
)
from blindtouch.env import BlindTouchEnv, EnvConfig


class PublicInfo(dict[str, Any]):
    """Fail loudly if a non-oracle controller attempts to inspect hidden state."""

    def __getitem__(self, key: str) -> Any:
        if key in {"object_params", "pad_forces", "lift_height"}:
            raise AssertionError(f"Controller attempted to read privileged key: {key}")
        return super().__getitem__(key)


def run_episode(
    env: BlindTouchEnv,
    controller: Any,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observation, info = env.reset(seed=seed, options=options)
    controller.reset(observation, info)
    while True:
        observation, _, terminated, truncated, info = env.step(controller.act(observation, info))
        if terminated or truncated:
            return info


def test_non_oracle_controllers_use_observation_and_internal_timing_only() -> None:
    observation = np.zeros(45, dtype=np.float32)
    public_info = PublicInfo({"phase": "explore", "step": 0})

    for controller in (
        FixedGripController(),
        ThresholdGripController(),
        ProbeThenLiftController(),
    ):
        controller.reset(observation, public_info)
        for _ in range(6):
            action = controller.act(observation, public_info)
            assert action.shape == (4,)
            assert action.dtype == np.float32


def test_demo_baselines_expose_failures_and_touch_adaptation() -> None:
    env = BlindTouchEnv(config=EnvConfig(exploration_steps=0, max_episode_steps=120))
    demo_objects = ("orange", "soap_bar", "tomato", "toy_car")

    fixed_outcomes = [
        run_episode(env, FixedGripController(), options={"demo_object": name})["outcome"]
        for name in demo_objects
    ]
    threshold_outcomes = [
        run_episode(env, ThresholdGripController(), options={"demo_object": name})["outcome"]
        for name in demo_objects
    ]
    probe_outcomes = [
        run_episode(env, ProbeThenLiftController(), options={"demo_object": name})["outcome"]
        for name in demo_objects
    ]

    assert "damage" in fixed_outcomes
    assert any(outcome != "success" for outcome in threshold_outcomes)
    assert probe_outcomes == ["success"] * len(demo_objects)
    env.close()


def test_oracle_debug_controller_certifies_named_feasible_suite() -> None:
    env = BlindTouchEnv(config=EnvConfig(exploration_steps=0, max_episode_steps=120))
    outcomes = [
        run_episode(env, OracleDebugController(), options={"demo_object": name})["outcome"]
        for name in ("orange", "soap_bar", "tomato", "toy_car")
    ]
    assert outcomes == ["success"] * 4
    env.close()


def test_training_archetype_baselines_smoke_run_is_nontrivial() -> None:
    env = BlindTouchEnv(config=EnvConfig(exploration_steps=0, max_episode_steps=120))
    counts: dict[str, Counter[str]] = {}
    for controller_type in (
        FixedGripController,
        ThresholdGripController,
        ProbeThenLiftController,
    ):
        outcomes = Counter(
            run_episode(env, controller_type(), seed=seed)["outcome"] for seed in range(20)
        )
        counts[controller_type.__name__] = outcomes

    assert sum(counts["FixedGripController"].values()) == 20
    assert counts["FixedGripController"]["damage"] > 0
    assert any(
        counts[name]["success"] > 0
        for name in ("ThresholdGripController", "ProbeThenLiftController")
    )
    env.close()
