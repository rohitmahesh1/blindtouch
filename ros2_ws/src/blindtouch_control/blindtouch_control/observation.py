"""Reconstruct BlindTouch policy observations from ROS topic state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sensor_msgs.msg import JointState

from blindtouch_interfaces.msg import ClawCommand, SimulationState, TactileState
from blindtouch_ros.constants import ACTION_NAMES, BASE_OBSERVATION_SIZE, POLICY_OBSERVATION_SIZE


FloatArray = NDArray[np.float32]


CTRL_LOW = np.array([-0.02, 0.0, 0.0, 0.0], dtype=np.float32)
CTRL_HIGH = np.array([0.14, 0.055, 0.055, 0.055], dtype=np.float32)
EFFORT_SCALE = np.array([80.0, 15.0, 15.0, 15.0], dtype=np.float32)


@dataclass
class ObservationBuilder:
    """Build the same 45-value frame used by BlindTouchEnv._observation()."""

    tactile_force_scale: float = 5.0
    velocity_scale: float = 0.5
    max_episode_steps: int = 120
    history_length: int = 8
    joint_positions: FloatArray = field(
        default_factory=lambda: CTRL_LOW.copy()
    )
    joint_velocities: FloatArray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    efforts: FloatArray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    taxel_forces: FloatArray = field(default_factory=lambda: np.zeros(27, dtype=np.float32))
    previous_action: FloatArray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    phase_lift_allowed: float = 0.0
    step: int = 0

    def __post_init__(self) -> None:
        if self.tactile_force_scale <= 0.0:
            raise ValueError("tactile_force_scale must be positive")
        if self.velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be positive")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if self.history_length < 1:
            raise ValueError("history_length must be positive")
        self._frames = np.zeros((self.history_length, BASE_OBSERVATION_SIZE), dtype=np.float32)
        self._last_step: int | None = None
        self.has_joint_state = False
        self.has_tactile_state = False
        self.has_sim_state = False

    @property
    def ready(self) -> bool:
        return self.has_tactile_state

    @property
    def policy_ready(self) -> bool:
        return self.has_tactile_state and self.has_joint_state and self.has_sim_state

    def update_joint_state(self, msg: JointState) -> None:
        index_by_name = {name: index for index, name in enumerate(msg.name)}
        for action_index, name in enumerate(ACTION_NAMES):
            source_index = index_by_name.get(name)
            if source_index is None:
                continue
            if source_index < len(msg.position):
                self.joint_positions[action_index] = float(msg.position[source_index])
            if source_index < len(msg.velocity):
                self.joint_velocities[action_index] = float(msg.velocity[source_index])
            if source_index < len(msg.effort):
                self.efforts[action_index] = float(msg.effort[source_index])
        self.has_joint_state = True

    def update_tactile_state(self, msg: TactileState) -> None:
        self.taxel_forces = np.asarray(msg.taxel_forces, dtype=np.float32).reshape(27)
        self.has_tactile_state = True

    def update_sim_state(self, msg: SimulationState) -> bool:
        reset_detected = self._last_step is not None and int(msg.step) < self._last_step
        self._last_step = int(msg.step)
        self.step = int(msg.step)
        self.phase_lift_allowed = 1.0 if msg.phase == "lift" else 0.0
        self.has_sim_state = True
        if reset_detected:
            self.reset_history()
        return reset_detected

    def set_previous_action(self, action: FloatArray) -> None:
        self.previous_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    def base_observation(self) -> FloatArray:
        joint_pos = 2.0 * (self.joint_positions - CTRL_LOW) / (CTRL_HIGH - CTRL_LOW) - 1.0
        joint_vel = np.clip(self.joint_velocities / self.velocity_scale, -1.0, 1.0)
        efforts = np.clip(self.efforts / EFFORT_SCALE, -1.0, 1.0)
        taxels = np.clip(self.taxel_forces / self.tactile_force_scale, 0.0, 1.0)
        phase = np.array(
            [
                self.phase_lift_allowed,
                1.0 - self.step / self.max_episode_steps,
            ],
            dtype=np.float32,
        )
        observation = np.concatenate(
            (joint_pos, joint_vel, efforts, taxels, self.previous_action, phase)
        ).astype(np.float32)
        return np.clip(observation, -1.0, 1.0)

    def stacked_observation(self) -> FloatArray:
        frame = self.base_observation()
        self._frames[:-1] = self._frames[1:].copy()
        self._frames[-1] = frame
        return self._frames.reshape(POLICY_OBSERVATION_SIZE).copy()

    def reset_history(self) -> None:
        frame = self.base_observation()
        self._frames[:] = frame


def action_from_command(msg: ClawCommand) -> FloatArray:
    return np.clip(
        np.array(
            [
                msg.palm_lift,
                msg.finger_1_close,
                msg.finger_2_close,
                msg.finger_3_close,
            ],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    )


def command_from_action(action: FloatArray, stamp: object | None = None) -> ClawCommand:
    clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    msg = ClawCommand()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.palm_lift = float(clipped[0])
    msg.finger_1_close = float(clipped[1])
    msg.finger_2_close = float(clipped[2])
    msg.finger_3_close = float(clipped[3])
    return msg
