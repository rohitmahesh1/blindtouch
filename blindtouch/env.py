"""
Gymnasium environment for the BlindTouch tactile lifting task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class ObjectParams:
    """Hidden physical properties for one rigid-cylinder episode."""

    radius: float
    half_height: float
    mass: float
    friction: float
    safe_force: float
    x_offset: float
    y_offset: float
    yaw: float


@dataclass(frozen=True)
class EnvConfig:
    """Task and randomization settings kept small enough to audit directly."""

    physics_steps_per_action: int = 20
    palm_target_delta: float = 0.004
    finger_target_delta: float = 0.003
    exploration_steps: int = 30
    max_episode_steps: int = 120
    success_hold_steps: int = 5
    settle_steps: int = 50
    lift_target_height: float = 0.045
    attempted_lift_height: float = 0.018
    contact_force_threshold: float = 0.02
    tactile_force_scale: float = 5.0
    velocity_scale: float = 0.5
    radius_range: tuple[float, float] = (0.023, 0.028)
    half_height_range: tuple[float, float] = (0.023, 0.037)
    mass_range: tuple[float, float] = (0.04, 0.18)
    friction_range: tuple[float, float] = (0.35, 1.20)
    safe_force_range: tuple[float, float] = (0.55, 2.20)
    offset_range: tuple[float, float] = (-0.004, 0.004)


class BlindTouchEnv(gym.Env[FloatArray, FloatArray]):
    """Touch-only mystery-object lifting environment backed by MuJoCo."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 25}

    ACTION_NAMES = ("palm_lift", "finger_1_close", "finger_2_close", "finger_3_close")
    TAXEL_NAMES = tuple(
        f"finger_{finger}_taxel_r{row}_c{column}_force"
        for finger in range(1, 4)
        for row in range(3)
        for column in range(3)
    )
    PAD_FORCE_NAMES = tuple(f"finger_{finger}_pad_force" for finger in range(1, 4))
    JOINT_POS_NAMES = (
        "palm_lift_pos",
        "finger_1_pos",
        "finger_2_pos",
        "finger_3_pos",
    )
    JOINT_VEL_NAMES = (
        "palm_lift_vel",
        "finger_1_vel",
        "finger_2_vel",
        "finger_3_vel",
    )
    EFFORT_NAMES = (
        "palm_lift_effort",
        "finger_1_effort",
        "finger_2_effort",
        "finger_3_effort",
    )
    OBSERVATION_LAYOUT = {
        "joint_position": slice(0, 4),
        "joint_velocity": slice(4, 8),
        "actuator_effort": slice(8, 12),
        "taxels": slice(12, 39),
        "previous_action": slice(39, 43),
        "phase": slice(43, 45),
    }

    def __init__(
        self,
        *,
        xml_path: str | Path | None = None,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        super().__init__()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render mode: {render_mode!r}")

        self.config = config or EnvConfig()
        if self.config.exploration_steps >= self.config.max_episode_steps:
            raise ValueError("exploration_steps must be smaller than max_episode_steps")
        if self.config.physics_steps_per_action < 1:
            raise ValueError("physics_steps_per_action must be positive")

        self.render_mode = render_mode
        self._width = width
        self._height = height
        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None

        asset_path = Path(xml_path) if xml_path else Path(__file__).with_name("assets") / "claw.xml"
        self.model = mujoco.MjModel.from_xml_path(str(asset_path))
        self.data = mujoco.MjData(self.model)

        self._object_body_id = self.model.body("object").id
        self._object_geom_id = self.model.geom("object_geom").id
        self._object_joint_id = self.model.joint("object_free").id
        self._object_qpos_adr = int(self.model.jnt_qposadr[self._object_joint_id])
        self._palm_qpos_adr = int(self.model.jnt_qposadr[self.model.joint("palm_lift").id])
        self._sensor_adrs = {
            name: int(self.model.sensor_adr[self.model.sensor(name).id])
            for name in (
                *self.JOINT_POS_NAMES,
                *self.JOINT_VEL_NAMES,
                *self.EFFORT_NAMES,
                *self.PAD_FORCE_NAMES,
                *self.TAXEL_NAMES,
            )
        }

        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32).copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32).copy()
        self._effort_scale = np.max(np.abs(self.model.actuator_forcerange), axis=1).astype(
            np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(45,), dtype=np.float32)

        self.object_params: ObjectParams | None = None
        self._step_count = 0
        self._successful_hold_steps = 0
        self._slip_events = 0
        self._peak_pad_force = 0.0
        self._rest_object_height = 0.0
        self._previous_lift_height = 0.0
        self._previous_action = np.zeros(4, dtype=np.float32)
        self._control_targets = self._ctrl_low.copy()
        self._outcome: str | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Reset into an open-claw episode with a newly hidden object."""

        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.object_params = self._choose_object_params(options or {})
        self._apply_object_params(self.object_params)
        self._place_object(self.object_params)
        self.data.ctrl[:] = self._ctrl_low
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.config.settle_steps):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        self._successful_hold_steps = 0
        self._slip_events = 0
        self._peak_pad_force = 0.0
        self._rest_object_height = self._object_height()
        self._previous_lift_height = 0.0
        self._previous_action.fill(0.0)
        self._control_targets = self._ctrl_low.copy()
        self._outcome = None

        observation = self._observation()
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(
        self, action: FloatArray
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by one controller command interval."""

        if self.object_params is None:
            raise RuntimeError("Call reset() before step().")

        policy_action = np.asarray(action, dtype=np.float32)
        if policy_action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {policy_action.shape}")
        policy_action = np.clip(policy_action, -1.0, 1.0)
        lift_allowed = self._step_count >= self.config.exploration_steps
        delta = policy_action * np.array(
            [
                self.config.palm_target_delta,
                self.config.finger_target_delta,
                self.config.finger_target_delta,
                self.config.finger_target_delta,
            ],
            dtype=np.float32,
        )
        if not lift_allowed:
            delta[0] = 0.0
        self._control_targets = np.clip(
            self._control_targets + delta, self._ctrl_low, self._ctrl_high
        )

        self.data.ctrl[:] = self._control_targets
        for _ in range(self.config.physics_steps_per_action):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        self._previous_action = policy_action.copy()

        pad_forces = self._pad_forces()
        max_force = float(np.max(pad_forces))
        self._peak_pad_force = max(self._peak_pad_force, max_force)
        lift_height = self._lift_height()
        object_falling = lift_height < self._previous_lift_height - 0.001
        contacting = int(np.count_nonzero(pad_forces > self.config.contact_force_threshold))
        palm_has_lifted = self.data.qpos[self._palm_qpos_adr] > self.config.attempted_lift_height
        slipped = bool(lift_allowed and palm_has_lifted and object_falling and contacting < 2)
        if slipped:
            self._slip_events += 1

        damaged = max_force > self.object_params.safe_force
        dropped = bool(
            lift_allowed
            and palm_has_lifted
            and contacting < 2
            and lift_height < 0.004
        )
        at_target = lift_height >= self.config.lift_target_height and contacting >= 2
        self._successful_hold_steps = self._successful_hold_steps + 1 if at_target else 0
        succeeded = self._successful_hold_steps >= self.config.success_hold_steps

        reward = self._reward(
            action=policy_action,
            pad_forces=pad_forces,
            lift_height=lift_height,
            slipped=slipped,
            damaged=damaged,
            dropped=dropped,
            succeeded=succeeded,
            lift_allowed=lift_allowed,
        )
        if damaged:
            self._outcome = "damage"
        elif succeeded:
            self._outcome = "success"
        elif dropped:
            self._outcome = "drop"

        terminated = bool(damaged or succeeded or dropped)
        truncated = bool(not terminated and self._step_count >= self.config.max_episode_steps)
        if truncated:
            self._outcome = "timeout"

        self._previous_lift_height = lift_height
        observation = self._observation()
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def render(self) -> NDArray[np.uint8] | None:
        """Render the scene for demonstrations, never as a policy observation."""

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=self._height, width=self._width)
            self._renderer.update_scene(self.data, camera="overview")
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mujoco_viewer

                self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _choose_object_params(self, options: Mapping[str, Any]) -> ObjectParams:
        provided = options.get("object_params", {})
        if provided is None:
            provided = {}
        if not isinstance(provided, Mapping):
            raise TypeError("options['object_params'] must be a mapping")

        def sampled(name: str, limits: tuple[float, float]) -> float:
            return float(provided[name]) if name in provided else float(self.np_random.uniform(*limits))

        return ObjectParams(
            radius=sampled("radius", self.config.radius_range),
            half_height=sampled("half_height", self.config.half_height_range),
            mass=sampled("mass", self.config.mass_range),
            friction=sampled("friction", self.config.friction_range),
            safe_force=sampled("safe_force", self.config.safe_force_range),
            x_offset=sampled("x_offset", self.config.offset_range),
            y_offset=sampled("y_offset", self.config.offset_range),
            yaw=sampled("yaw", (-np.pi, np.pi)),
        )

    def _apply_object_params(self, params: ObjectParams) -> None:
        if params.radius <= 0 or params.half_height <= 0 or params.mass <= 0:
            raise ValueError("Object radius, height, and mass must be positive")
        if params.friction <= 0 or params.safe_force <= 0:
            raise ValueError("Object friction and safe_force must be positive")

        self.model.geom_size[self._object_geom_id, :2] = (params.radius, params.half_height)
        self.model.geom_friction[self._object_geom_id, 0] = params.friction
        self.model.body_mass[self._object_body_id] = params.mass
        full_height = 2.0 * params.half_height
        transverse_inertia = params.mass * (3.0 * params.radius**2 + full_height**2) / 12.0
        axial_inertia = 0.5 * params.mass * params.radius**2
        self.model.body_inertia[self._object_body_id] = (
            transverse_inertia,
            transverse_inertia,
            axial_inertia,
        )

    def _place_object(self, params: ObjectParams) -> None:
        half_yaw = 0.5 * params.yaw
        qpos = self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 7]
        qpos[:] = (
            params.x_offset,
            params.y_offset,
            params.half_height + 0.001,
            np.cos(half_yaw),
            0.0,
            0.0,
            np.sin(half_yaw),
        )

    def _reward(
        self,
        *,
        action: FloatArray,
        pad_forces: FloatArray,
        lift_height: float,
        slipped: bool,
        damaged: bool,
        dropped: bool,
        succeeded: bool,
        lift_allowed: bool,
    ) -> float:
        safe_force = self.object_params.safe_force if self.object_params else 1.0
        force_fraction = float(np.max(pad_forces) / safe_force)
        reward = -0.005
        reward -= 0.002 * float(np.mean(np.square(action)))
        reward -= 0.006 * force_fraction**2
        if lift_allowed:
            reward += 4.0 * (lift_height - self._previous_lift_height) / self.config.lift_target_height
        if slipped:
            reward -= 0.25
        if damaged:
            reward -= 10.0
        elif dropped:
            reward -= 5.0
        elif succeeded:
            reward += 10.0
        return reward

    def _observation(self) -> FloatArray:
        joint_pos = self._read_many(self.JOINT_POS_NAMES)
        joint_pos = 2.0 * (joint_pos - self._ctrl_low) / (self._ctrl_high - self._ctrl_low) - 1.0
        joint_vel = np.clip(
            self._read_many(self.JOINT_VEL_NAMES) / self.config.velocity_scale, -1.0, 1.0
        )
        efforts = np.clip(self._read_many(self.EFFORT_NAMES) / self._effort_scale, -1.0, 1.0)
        taxels = np.clip(
            self._read_many(self.TAXEL_NAMES) / self.config.tactile_force_scale, 0.0, 1.0
        )
        phase = np.array(
            [
                float(self._step_count >= self.config.exploration_steps),
                1.0 - self._step_count / self.config.max_episode_steps,
            ],
            dtype=np.float32,
        )
        observation = np.concatenate(
            (joint_pos, joint_vel, efforts, taxels, self._previous_action, phase)
        ).astype(np.float32)
        return np.clip(observation, -1.0, 1.0)

    def _info(self) -> dict[str, Any]:
        params = asdict(self.object_params) if self.object_params else {}
        pad_forces = self._pad_forces()
        return {
            "phase": "lift" if self._step_count >= self.config.exploration_steps else "explore",
            "step": self._step_count,
            "outcome": self._outcome,
            "object_params": params,
            "object_height": self._object_height(),
            "lift_height": self._lift_height(),
            "control_targets": self._control_targets.copy(),
            "pad_forces": pad_forces.copy(),
            "peak_pad_force": self._peak_pad_force,
            "slip_events": self._slip_events,
        }

    def _read_many(self, names: tuple[str, ...]) -> FloatArray:
        return np.array([self.data.sensordata[self._sensor_adrs[name]] for name in names], dtype=np.float32)

    def _pad_forces(self) -> FloatArray:
        return self._read_many(self.PAD_FORCE_NAMES)

    def _object_height(self) -> float:
        return float(self.data.xpos[self._object_body_id, 2])

    def _lift_height(self) -> float:
        return max(0.0, self._object_height() - self._rest_object_height)


__all__ = ["BlindTouchEnv", "EnvConfig", "ObjectParams"]
