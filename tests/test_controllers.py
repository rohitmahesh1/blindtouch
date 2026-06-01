from collections import Counter
from typing import Any

import numpy as np

from blindtouch.controllers import (
    FixedGripController,
    ProbeThenLiftController,
    PrivilegedFeasibilityController,
    SafeForceGripController,
    ThresholdGripController,
    per_finger_max_taxel_force,
)
from blindtouch.env import BlindTouchEnv, EnvConfig


REFERENCE_OBJECT = {
    "shape": "cylinder",
    "half_size_x": 0.024,
    "half_size_y": 0.024,
    "half_size_z": 0.030,
    "mass": 0.10,
    "friction": 1.0,
    "safe_force": 2.0,
    "x_offset": 0.0,
    "y_offset": 0.0,
    "yaw": 0.0,
}


class PublicInfo(dict[str, Any]):
    """Fail loudly if a public controller attempts to inspect hidden state."""

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


def test_public_controllers_use_observation_and_internal_timing_only() -> None:
    observation = np.zeros(45, dtype=np.float32)
    public_info = PublicInfo({"phase": "explore", "step": 0})

    for controller in (
        FixedGripController(),
        SafeForceGripController(),
        ThresholdGripController(),
        ProbeThenLiftController(),
    ):
        controller.reset(observation, public_info)
        for _ in range(6):
            action = controller.act(observation, public_info)
            assert action.shape == (4,)
            assert action.dtype == np.float32


def test_safe_force_controller_balances_individual_fingertip_taxels() -> None:
    observation = np.zeros(45, dtype=np.float32)
    controller = SafeForceGripController()
    controller.reset(observation, {"phase": "explore", "step": 0})

    action = controller.act(observation, {"phase": "explore", "step": 0})
    assert action[0] == 0.0
    assert np.all(action[1:] > 0.0)

    observation[12] = 0.80 / 5.0
    observation[21] = 0.10 / 5.0
    observation[30] = 0.50 / 5.0
    forces = per_finger_max_taxel_force(observation)
    np.testing.assert_allclose(forces, [0.80, 0.10, 0.50], atol=1e-6)

    action = controller.act(observation, {"phase": "explore", "step": 1})
    assert action[1] < 0.0
    assert action[2] > 0.0
    assert action[3] == 0.0


def test_privileged_feasibility_controller_can_certify_a_reference_object() -> None:
    env = BlindTouchEnv(config=EnvConfig(exploration_steps=0, max_episode_steps=120))
    result = run_episode(
        env,
        PrivilegedFeasibilityController(scripted_close_phases=((0.70, 30),)),
        options={"object_params": REFERENCE_OBJECT},
    )
    assert result["outcome"] == "success"
    env.close()


def test_training_archetype_baselines_cover_distinct_outcomes() -> None:
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
