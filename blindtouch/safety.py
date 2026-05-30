"""Observable tactile safety filters for BlindTouch policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .controllers import DEFAULT_TACTILE_FORCE_SCALE, per_finger_max_taxel_force


Action = NDArray[np.float32]
Observation = NDArray[np.float32]
OBSERVATION_FRAME_SIZE: Final[int] = 45


@dataclass(frozen=True)
class TactileSafetyFilterConfig:
    """Force/rate guard settings for touch-only policy actions.

    The filter is intentionally object-agnostic. It consumes the same tactile
    observation stream available to the policy and keeps only short internal
    history for force-rate and action smoothing.
    """

    name: str
    soft_force: float
    high_force: float
    force_rate: float
    release_action: float
    close_cap: float
    high_force_lift_cap: float
    precontact_lift_cap: float
    min_lift_contacts: int
    ready_force: float
    stable_steps: int
    contact_threshold: float = 0.015
    smooth_alpha: float = 1.0
    lift_cap: float | None = None
    tactile_force_scale: float = DEFAULT_TACTILE_FORCE_SCALE

    def __post_init__(self) -> None:
        if self.soft_force < 0.0 or self.high_force <= 0.0:
            raise ValueError("force thresholds must be nonnegative with positive high_force")
        if self.soft_force > self.high_force:
            raise ValueError("soft_force must be no greater than high_force")
        if self.force_rate < 0.0:
            raise ValueError("force_rate must be nonnegative")
        if self.min_lift_contacts < 0 or self.stable_steps < 0:
            raise ValueError("contact and stability counts must be nonnegative")
        if not 0.0 < self.smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be in (0, 1]")
        if self.tactile_force_scale <= 0.0:
            raise ValueError("tactile_force_scale must be positive")


@dataclass(frozen=True)
class SafetyFilterResult:
    """One safety-filter decision for logging or residual-head training."""

    action: Action
    raw_action: Action
    intervened: bool
    max_force: float
    max_force_rate: float
    contact_count: int
    ready_count: int
    stable_steps: int

    @property
    def residual(self) -> Action:
        return (self.action - self.raw_action).astype(np.float32)


def fragile_smooth_0p32_config() -> TactileSafetyFilterConfig:
    """Current best fragile safety wrapper from the scratch guard sweep."""

    return TactileSafetyFilterConfig(
        name="fragile_smooth_0p32",
        soft_force=0.25,
        high_force=0.32,
        force_rate=0.075,
        release_action=-0.28,
        close_cap=0.0,
        high_force_lift_cap=0.0,
        precontact_lift_cap=0.0,
        min_lift_contacts=2,
        ready_force=0.060,
        stable_steps=2,
        smooth_alpha=0.55,
        lift_cap=0.55,
    )


class TactileSafetyFilter:
    """Clamp force-unsafe policy commands using public tactile observations."""

    def __init__(self, config: TactileSafetyFilterConfig | None = None) -> None:
        self.config = config or fragile_smooth_0p32_config()
        self.previous_forces = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.stable_steps = 0

    def reset(self) -> None:
        self.previous_forces[:] = 0.0
        self.previous_action[:] = 0.0
        self.stable_steps = 0

    def apply(self, action: Action, observation: Observation) -> SafetyFilterResult:
        config = self.config
        raw_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        guarded = raw_action.copy()
        forces = _latest_per_finger_forces(observation, config.tactile_force_scale)
        force_rate = forces - self.previous_forces
        contact_count = int(np.count_nonzero(forces >= config.contact_threshold))
        ready_count = int(np.count_nonzero(forces >= config.ready_force))
        max_force = float(np.max(forces))
        max_rate = float(np.max(force_rate))

        balanced = ready_count >= config.min_lift_contacts and max_force < config.high_force
        self.stable_steps = self.stable_steps + 1 if balanced else 0

        for finger_index, force in enumerate(forces):
            action_index = finger_index + 1
            if force >= config.high_force or force_rate[finger_index] >= config.force_rate:
                guarded[action_index] = min(guarded[action_index], config.release_action)
            elif force >= config.soft_force:
                guarded[action_index] = min(guarded[action_index], config.close_cap)

        if max_force >= config.high_force or max_rate >= config.force_rate:
            guarded[0] = min(guarded[0], config.high_force_lift_cap)
        elif contact_count < config.min_lift_contacts or self.stable_steps < config.stable_steps:
            guarded[0] = min(guarded[0], config.precontact_lift_cap)

        if config.lift_cap is not None:
            guarded[0] = min(guarded[0], config.lift_cap)

        if config.smooth_alpha < 1.0:
            guarded = self.previous_action + config.smooth_alpha * (guarded - self.previous_action)

        filtered_action = np.clip(guarded.astype(np.float32), -1.0, 1.0)
        result = SafetyFilterResult(
            action=filtered_action,
            raw_action=raw_action,
            intervened=bool(np.any(np.abs(filtered_action - raw_action) > 1e-6)),
            max_force=max_force,
            max_force_rate=max_rate,
            contact_count=contact_count,
            ready_count=ready_count,
            stable_steps=self.stable_steps,
        )
        self.previous_forces = forces.copy()
        self.previous_action = filtered_action.copy()
        return result


def _latest_per_finger_forces(
    observation: Observation, tactile_force_scale: float
) -> NDArray[np.float32]:
    flattened = np.asarray(observation, dtype=np.float32).reshape(-1)
    if flattened.size % OBSERVATION_FRAME_SIZE != 0:
        raise ValueError(
            f"observation length must be a multiple of {OBSERVATION_FRAME_SIZE}, got {flattened.size}"
        )
    frame = flattened[-OBSERVATION_FRAME_SIZE:]
    return per_finger_max_taxel_force(frame, tactile_force_scale)


__all__ = [
    "SafetyFilterResult",
    "TactileSafetyFilter",
    "TactileSafetyFilterConfig",
    "fragile_smooth_0p32_config",
]
