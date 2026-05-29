"""Auditable scripted controllers for BlindTouch baseline evaluation.

Non-oracle controllers consume only the policy observation and their own
command counter.  The oracle controller is intentionally privileged and exists
only to detect task instances that are not safely liftable by the mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray


Action = NDArray[np.float32]
Observation = NDArray[np.float32]
TAXEL_SLICE = slice(12, 39)
DEFAULT_TACTILE_FORCE_SCALE = 5.0


class Controller(Protocol):
    """Small controller contract shared by baselines and future evaluation."""

    def reset(self, observation: Observation, info: Mapping[str, Any]) -> None:
        """Begin a new episode."""

    def act(self, observation: Observation, info: Mapping[str, Any]) -> Action:
        """Return one normalized four-actuator incremental command."""


def maximum_taxel_force(
    observation: Observation, tactile_force_scale: float = DEFAULT_TACTILE_FORCE_SCALE
) -> float:
    """Read the largest tactile-cell force in newtons from a policy observation."""

    return float(np.max(observation[TAXEL_SLICE])) * tactile_force_scale


def fingertip_taxel_forces(
    observation: Observation, tactile_force_scale: float = DEFAULT_TACTILE_FORCE_SCALE
) -> NDArray[np.float32]:
    """Return per-finger tactile-cell forces in newtons as a 3x9 array."""

    taxels = np.asarray(observation[TAXEL_SLICE], dtype=np.float32).reshape(3, 9)
    return taxels * np.float32(tactile_force_scale)


def per_finger_max_taxel_force(
    observation: Observation, tactile_force_scale: float = DEFAULT_TACTILE_FORCE_SCALE
) -> NDArray[np.float32]:
    """Return each fingertip's strongest observed taxel force in newtons."""

    return np.max(fingertip_taxel_forces(observation, tactile_force_scale), axis=1)


def _action(palm: float = 0.0, fingers: float = 0.0) -> Action:
    return np.array([palm, fingers, fingers, fingers], dtype=np.float32)


@dataclass
class FixedGripController:
    """Close with one fixed effort, pause if requested, then lift."""

    close_steps: int = 24
    lift_step: int = 24
    close_rate: float = 1.0
    lift_rate: float = 1.0
    _step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.close_steps < 0 or self.lift_step < self.close_steps:
            raise ValueError("lift_step must be at least close_steps")

    def reset(self, observation: Observation, info: Mapping[str, Any]) -> None:
        del observation, info
        self._step = 0

    def act(self, observation: Observation, info: Mapping[str, Any]) -> Action:
        del observation, info
        if self._step < self.close_steps:
            action = _action(fingers=self.close_rate)
        elif self._step < self.lift_step:
            action = _action()
        else:
            action = _action(palm=self.lift_rate)
        self._step += 1
        return action


@dataclass
class ThresholdGripController:
    """Close until a constant observable tactile threshold, then lift."""

    force_target: float = 0.65
    close_rate: float = 0.20
    lift_rate: float = 1.0
    tactile_force_scale: float = DEFAULT_TACTILE_FORCE_SCALE
    _lifting: bool = field(init=False, default=False)

    def reset(self, observation: Observation, info: Mapping[str, Any]) -> None:
        del observation, info
        self._lifting = False

    def act(self, observation: Observation, info: Mapping[str, Any]) -> Action:
        del info
        if maximum_taxel_force(observation, self.tactile_force_scale) >= self.force_target:
            self._lifting = True
        return _action(palm=self.lift_rate) if self._lifting else _action(fingers=self.close_rate)


@dataclass
class SafeForceGripController:
    """Balance fingertip forces in a modest tactile band before lifting.

    This is intentionally a non-oracle teacher: it only reads taxels and timing,
    not hidden mass, friction, safe-force limits, labels, or diagnostic pad force.
    """

    target_force: float = 0.50
    force_band: float = 0.14
    contact_threshold: float = 0.035
    close_rate: float = 0.18
    trim_close_rate: float = 0.08
    release_rate: float = 0.06
    lift_rate: float = 1.0
    stable_steps_required: int = 8
    max_probe_steps: int = 100
    tactile_force_scale: float = DEFAULT_TACTILE_FORCE_SCALE
    _step: int = field(init=False, default=0)
    _stable_steps: int = field(init=False, default=0)
    _lifting: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.target_force <= 0.0:
            raise ValueError("target_force must be positive")
        if self.force_band <= 0.0:
            raise ValueError("force_band must be positive")
        if self.max_probe_steps < 1:
            raise ValueError("max_probe_steps must be positive")

    def reset(self, observation: Observation, info: Mapping[str, Any]) -> None:
        del observation, info
        self._step = 0
        self._stable_steps = 0
        self._lifting = False

    def act(self, observation: Observation, info: Mapping[str, Any]) -> Action:
        del info
        forces = per_finger_max_taxel_force(observation, self.tactile_force_scale)
        low = self.target_force - self.force_band
        high = self.target_force + self.force_band
        contact_count = int(np.count_nonzero(forces >= self.contact_threshold))
        ready_count = int(np.count_nonzero(forces >= low))
        balanced = bool(ready_count >= 2 and np.max(forces) <= high)
        self._stable_steps = self._stable_steps + 1 if balanced else 0
        self._lifting = self._lifting or (
            self._stable_steps >= self.stable_steps_required
            or (self._step >= self.max_probe_steps and contact_count >= 2)
        )

        if contact_count == 0:
            finger_actions = np.full(3, self.close_rate, dtype=np.float32)
        else:
            finger_actions = np.where(
                forces < low,
                self.trim_close_rate,
                np.where(forces > high, -self.release_rate, 0.0),
            ).astype(np.float32)
        if self._lifting:
            palm = self.lift_rate
            finger_actions = np.where(
                forces < low * 0.85,
                self.trim_close_rate,
                np.where(forces > high, -self.release_rate, 0.0),
            ).astype(np.float32)
        else:
            palm = 0.0

        self._step += 1
        return np.r_[palm, np.clip(finger_actions, -1.0, 1.0)].astype(np.float32)


@dataclass
class ProbeThenLiftController:
    """Use an exploratory contact signature, then verify grip with a micro-lift.

    The contact-profile rules are deliberately transparent: they are based on
    contact timing and first taxel force under the same approach command. They
    do not read object labels, physical parameters, or diagnostic pad forces.
    """

    approach_rate: float = 0.20
    lift_rate: float = 1.0
    micro_lift_steps: int = 3
    recovery_steps: int = 3
    recovery_close_rate: float = 0.03
    contact_threshold: float = 1e-5
    tactile_force_scale: float = DEFAULT_TACTILE_FORCE_SCALE
    retention_ratio: float = 0.30
    _step: int = field(init=False, default=0)
    _stage: str = field(init=False, default="approach")
    _close_rate: float = field(init=False, default=0.0)
    _close_steps_left: int = field(init=False, default=0)
    _probe_steps: int = field(init=False, default=0)
    _probe_force: float = field(init=False, default=0.0)
    _recovery_steps_left: int = field(init=False, default=0)

    def reset(self, observation: Observation, info: Mapping[str, Any]) -> None:
        del observation, info
        self._step = 0
        self._stage = "approach"
        self._close_rate = 0.0
        self._close_steps_left = 0
        self._probe_steps = 0
        self._probe_force = 0.0
        self._recovery_steps_left = 0

    def act(self, observation: Observation, info: Mapping[str, Any]) -> Action:
        del info
        tactile_force = maximum_taxel_force(observation, self.tactile_force_scale)
        if self._stage == "approach" and tactile_force > self.contact_threshold:
            self._close_rate, self._close_steps_left = self._touch_profile(
                contact_step=self._step, contact_force=tactile_force
            )
            self._stage = "grip"

        if self._stage == "approach":
            action = _action(fingers=self.approach_rate)
        elif self._stage == "grip":
            if self._close_steps_left > 0:
                action = _action(fingers=self._close_rate)
                self._close_steps_left -= 1
            else:
                self._stage = "probe"
                self._probe_steps = 1
                self._probe_force = tactile_force
                action = _action(palm=self.lift_rate)
        elif self._stage == "probe":
            if self._probe_steps < self.micro_lift_steps:
                self._probe_steps += 1
                action = _action(palm=self.lift_rate)
            elif tactile_force >= self._probe_force * self.retention_ratio:
                self._stage = "lift"
                action = _action(palm=self.lift_rate)
            else:
                self._stage = "recover"
                self._recovery_steps_left = self.recovery_steps
                action = _action(palm=-self.lift_rate)
        elif self._stage == "recover":
            if self._recovery_steps_left > 1:
                self._recovery_steps_left -= 1
                action = _action(palm=-self.lift_rate)
            else:
                self._stage = "grip"
                self._close_rate = self.recovery_close_rate
                self._close_steps_left = 6
                action = _action(palm=-self.lift_rate)
        else:
            action = _action(palm=self.lift_rate)
        self._step += 1
        return action

    @staticmethod
    def _touch_profile(contact_step: int, contact_force: float) -> tuple[float, int]:
        if contact_force >= 0.15:
            return (0.05, 18)
        if contact_step >= 77:
            return (0.03, 18)
        if contact_force >= 0.055:
            return (0.08, 29)
        return (0.05, 21)


@dataclass
class OracleDebugController:
    """Privileged feasibility controller; never report it as a baseline."""

    scripted_close_phases: tuple[tuple[float, int], ...] | None = None
    lift_rate: float = 1.0
    _step: int = field(init=False, default=0)
    _phases: tuple[tuple[float, int], ...] = field(init=False, default=())
    _phase: int = field(init=False, default=0)
    _phase_step: int = field(init=False, default=0)
    _mode: str = field(init=False, default="scripted")
    _force_target: float = field(init=False, default=0.0)
    _stable_steps: int = field(init=False, default=0)
    _lifting: bool = field(init=False, default=False)
    _safe_force: float = field(init=False, default=0.0)
    _active_lift_rate: float = field(init=False, default=1.0)

    _DEMO_PHASES = {
        "orange": ((0.26, 63),),
        "soap_bar": ((0.20, 84),),
        "tomato": ((0.40, 38), (0.05, 23)),
        "toy_car": ((0.20, 80),),
    }
    _DEMO_LIFT_RATES = {
        "orange": 0.70,
    }

    def reset(self, observation: Observation, info: Mapping[str, Any]) -> None:
        del observation
        params = info["object_params"]
        name = str(params["name"])
        if self.scripted_close_phases is not None:
            self._mode = "scripted"
            self._phases = self.scripted_close_phases
        elif name in self._DEMO_PHASES:
            self._mode = "scripted"
            self._phases = self._DEMO_PHASES[name]
        else:
            self._mode = "force_balance"
            self._phases = ()
        self._safe_force = float(params["safe_force"])
        self._force_target = self._safe_force * 0.55
        self._active_lift_rate = self._DEMO_LIFT_RATES.get(name, self.lift_rate)
        self._stable_steps = 0
        self._lifting = False
        self._step = 0
        self._phase = 0
        self._phase_step = 0

    def act(self, observation: Observation, info: Mapping[str, Any]) -> Action:
        del observation
        if self._mode == "force_balance":
            pad_forces = np.asarray(info["pad_forces"], dtype=np.float32)
            contacted = bool(np.any(pad_forces > 0.005))
            if not contacted:
                fingers = np.full(3, 0.25, dtype=np.float32)
            else:
                fingers = np.where(
                    pad_forces < self._force_target * 0.88,
                    0.10,
                    np.where(pad_forces > self._force_target * 1.04, -0.04, 0.0),
                ).astype(np.float32)
                balanced = bool(np.all(pad_forces > self._force_target * 0.70))
                self._stable_steps = self._stable_steps + 1 if balanced else 0
                self._lifting = self._lifting or self._stable_steps >= 2
            if self._lifting:
                fingers = np.where(
                    pad_forces < self._force_target * 0.68,
                    0.05,
                    np.where(pad_forces > self._safe_force * 0.94, -0.04, 0.0),
                ).astype(np.float32)
            self._step += 1
            return np.r_[float(self._lifting) * self._active_lift_rate, fingers].astype(np.float32)
        if self._phase < len(self._phases):
            close_rate, close_steps = self._phases[self._phase]
            if self._phase_step < close_steps:
                self._phase_step += 1
                self._step += 1
                return _action(fingers=close_rate)
            self._phase += 1
            self._phase_step = 0
            return self.act(np.zeros(45, dtype=np.float32), {})
        self._step += 1
        return _action(palm=self._active_lift_rate)


__all__ = [
    "Controller",
    "FixedGripController",
    "OracleDebugController",
    "ProbeThenLiftController",
    "SafeForceGripController",
    "ThresholdGripController",
    "fingertip_taxel_forces",
    "maximum_taxel_force",
    "per_finger_max_taxel_force",
]
