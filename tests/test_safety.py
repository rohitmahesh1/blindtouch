import numpy as np
import pytest

from blindtouch.safety import TactileSafetyFilter


def test_tactile_safety_filter_delays_lift_until_contact_is_stable() -> None:
    safety_filter = TactileSafetyFilter()
    observation = np.zeros(360, dtype=np.float32)
    raw_action = np.array([1.0, 0.2, 0.2, 0.2], dtype=np.float32)

    result = safety_filter.apply(raw_action, observation)

    assert result.intervened
    assert result.contact_count == 0
    assert result.action[0] == pytest.approx(0.0)


def test_tactile_safety_filter_releases_high_force_fingers() -> None:
    safety_filter = TactileSafetyFilter()
    observation = np.zeros(45, dtype=np.float32)
    observation[12] = 0.40 / 5.0
    observation[21] = 0.08 / 5.0
    observation[30] = 0.30 / 5.0
    raw_action = np.array([1.0, 0.5, 0.5, 0.5], dtype=np.float32)

    result = safety_filter.apply(raw_action, observation)

    assert result.intervened
    assert result.max_force == pytest.approx(0.40)
    assert result.action[0] == pytest.approx(0.0)
    assert result.action[1] < 0.0
    assert result.action[3] <= result.raw_action[3]


def test_tactile_safety_filter_reports_residual_from_public_observation() -> None:
    safety_filter = TactileSafetyFilter()
    observation = np.zeros(45, dtype=np.float32)
    raw_action = np.array([0.25, 0.1, 0.1, 0.1], dtype=np.float32)

    result = safety_filter.apply(raw_action, observation)

    np.testing.assert_allclose(result.residual, result.action - raw_action)
    assert result.ready_count == 0
    assert result.stable_steps == 0
