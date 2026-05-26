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
    """Hidden physical properties for one rigid primitive object episode."""

    shape: str
    half_size_x: float
    half_size_y: float
    half_size_z: float
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
    max_tilt_radians: float = np.deg2rad(40.0)
    slip_distance_threshold: float = 0.0005
    training_shapes: tuple[str, ...] = ("cylinder", "box", "capsule")
    radial_size_range: tuple[float, float] = (0.023, 0.031)
    capsule_radius_range: tuple[float, float] = (0.023, 0.028)
    half_height_range: tuple[float, float] = (0.022, 0.040)
    mass_range: tuple[float, float] = (0.03, 0.18)
    friction_range: tuple[float, float] = (0.20, 1.20)
    safe_force_range: tuple[float, float] = (0.40, 2.80)
    offset_range: tuple[float, float] = (-0.006, 0.006)
    nominal_pad_force_capacity: float = 0.70
    holding_force_margin: float = 0.80
    safe_force_margin: float = 1.35


class BlindTouchEnv(gym.Env[FloatArray, FloatArray]):
    """Touch-only mystery-object lifting environment backed by MuJoCo."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 25}

    ACTION_NAMES = ("palm_lift", "finger_1_close", "finger_2_close", "finger_3_close")
    OBJECT_SHAPES = ("cylinder", "box", "capsule", "ellipsoid")
    TRAINING_SHAPES = ("cylinder", "box", "capsule")
    GRASP_PAIR_NAMES = tuple(f"finger_{finger}_object_contact" for finger in range(1, 4))
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
        unknown_shapes = set(self.config.training_shapes) - set(self.OBJECT_SHAPES)
        if unknown_shapes:
            raise ValueError(f"Unsupported training shapes: {sorted(unknown_shapes)}")
        if not self.config.training_shapes:
            raise ValueError("training_shapes must contain at least one shape")

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
        self._grasp_pair_ids = np.array(
            [self.model.pair(name).id for name in self.GRASP_PAIR_NAMES], dtype=np.int32
        )
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
        self._previous_object_xy = np.zeros(2, dtype=np.float64)
        self._cumulative_slip_distance = 0.0
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
        self._previous_object_xy = self.data.xpos[self._object_body_id, :2].copy()
        self._cumulative_slip_distance = 0.0
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
        object_xy = self.data.xpos[self._object_body_id, :2].copy()
        lateral_distance = float(np.linalg.norm(object_xy - self._previous_object_xy))
        contacting = int(np.count_nonzero(pad_forces > self.config.contact_force_threshold))
        palm_has_lifted = self.data.qpos[self._palm_qpos_adr] > self.config.attempted_lift_height
        slipped = bool(
            lift_allowed
            and palm_has_lifted
            and (
                (object_falling and contacting < 2)
                or lateral_distance > self.config.slip_distance_threshold
            )
        )
        if slipped:
            self._slip_events += 1
            self._cumulative_slip_distance += lateral_distance

        damaged = max_force > self.object_params.safe_force
        dropped = bool(
            lift_allowed
            and palm_has_lifted
            and contacting < 2
            and lift_height < 0.004
        )
        tilt = self._object_tilt()
        unstable = bool(
            lift_allowed
            and palm_has_lifted
            and lift_height > 0.004
            and tilt > self.config.max_tilt_radians
        )
        at_target = lift_height >= self.config.lift_target_height and contacting >= 2
        self._successful_hold_steps = self._successful_hold_steps + 1 if at_target else 0
        succeeded = self._successful_hold_steps >= self.config.success_hold_steps and not unstable

        reward = self._reward(
            action=policy_action,
            pad_forces=pad_forces,
            lift_height=lift_height,
            slipped=slipped,
            damaged=damaged,
            dropped=dropped,
            unstable=unstable,
            succeeded=succeeded,
            lift_allowed=lift_allowed,
        )
        if damaged:
            self._outcome = "damage"
        elif succeeded:
            self._outcome = "success"
        elif unstable:
            self._outcome = "unstable"
        elif dropped:
            self._outcome = "drop"

        terminated = bool(damaged or succeeded or unstable or dropped)
        truncated = bool(not terminated and self._step_count >= self.config.max_episode_steps)
        if truncated:
            self._outcome = "timeout"

        self._previous_lift_height = lift_height
        self._previous_object_xy = object_xy
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

        shape = str(
            provided.get("shape", self.np_random.choice(self.config.training_shapes))
        )
        if shape not in self.OBJECT_SHAPES:
            raise ValueError(f"Unsupported object shape: {shape!r}")
        if shape == "capsule":
            half_size_x = sampled("half_size_x", self.config.capsule_radius_range)
            half_size_y = float(provided.get("half_size_y", half_size_x))
            minimum_height = half_size_x + 0.004
            if "half_size_z" in provided:
                half_size_z = float(provided["half_size_z"])
            else:
                half_size_z = float(
                    self.np_random.uniform(
                        max(self.config.half_height_range[0], minimum_height),
                        self.config.half_height_range[1],
                    )
                )
        elif shape == "cylinder":
            half_size_x = sampled("half_size_x", self.config.radial_size_range)
            half_size_y = float(provided.get("half_size_y", half_size_x))
            half_size_z = sampled("half_size_z", self.config.half_height_range)
        else:
            half_size_x = sampled("half_size_x", self.config.radial_size_range)
            half_size_y = sampled("half_size_y", self.config.radial_size_range)
            half_size_z = sampled("half_size_z", self.config.half_height_range)

        friction = sampled("friction", self.config.friction_range)
        if "mass" in provided:
            mass = float(provided["mass"])
        else:
            feasible_mass_ceiling = (
                3.0
                * friction
                * self.config.nominal_pad_force_capacity
                * self.config.holding_force_margin
                / 9.81
            )
            upper_mass = max(
                self.config.mass_range[0],
                min(self.config.mass_range[1], feasible_mass_ceiling),
            )
            mass = float(self.np_random.uniform(self.config.mass_range[0], upper_mass))
        required_pad_force = mass * 9.81 / (3.0 * friction)
        if "safe_force" in provided:
            safe_force = float(provided["safe_force"])
        else:
            lower_safe_force = max(
                self.config.safe_force_range[0],
                required_pad_force * self.config.safe_force_margin,
            )
            upper_safe_force = max(lower_safe_force, self.config.safe_force_range[1])
            safe_force = float(self.np_random.uniform(lower_safe_force, upper_safe_force))

        return ObjectParams(
            shape=shape,
            half_size_x=half_size_x,
            half_size_y=half_size_y,
            half_size_z=half_size_z,
            mass=mass,
            friction=friction,
            safe_force=safe_force,
            x_offset=sampled("x_offset", self.config.offset_range),
            y_offset=sampled("y_offset", self.config.offset_range),
            yaw=sampled("yaw", (-np.pi, np.pi)),
        )

    def _apply_object_params(self, params: ObjectParams) -> None:
        if min(params.half_size_x, params.half_size_y, params.half_size_z, params.mass) <= 0:
            raise ValueError("Object dimensions and mass must be positive")
        if params.friction <= 0 or params.safe_force <= 0:
            raise ValueError("Object friction and safe_force must be positive")
        if params.shape == "capsule" and params.half_size_z <= params.half_size_x:
            raise ValueError("Capsule half_size_z must exceed its radius")

        self.model.geom_type[self._object_geom_id] = self._geom_type(params.shape)
        self.model.geom_size[self._object_geom_id] = self._geom_size(params)
        self.model.geom_rbound[self._object_geom_id] = self._geom_rbound(params)
        self.model.geom_friction[self._object_geom_id, 0] = params.friction
        for pair_id in self._grasp_pair_ids:
            self.model.pair_friction[pair_id, :2] = params.friction
        self.model.body_mass[self._object_body_id] = params.mass
        self.model.body_inertia[self._object_body_id] = self._body_inertia(params)

    def _place_object(self, params: ObjectParams) -> None:
        half_yaw = 0.5 * params.yaw
        qpos = self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 7]
        qpos[:] = (
            params.x_offset,
            params.y_offset,
            params.half_size_z + 0.001,
            np.cos(half_yaw),
            0.0,
            0.0,
            np.sin(half_yaw),
        )

    @staticmethod
    def _geom_type(shape: str) -> int:
        return {
            "cylinder": int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            "box": int(mujoco.mjtGeom.mjGEOM_BOX),
            "capsule": int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            "ellipsoid": int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        }[shape]

    @staticmethod
    def _geom_size(params: ObjectParams) -> NDArray[np.float64]:
        if params.shape == "cylinder":
            return np.array([params.half_size_x, params.half_size_z, 0.0])
        if params.shape == "capsule":
            cylinder_half_length = params.half_size_z - params.half_size_x
            return np.array([params.half_size_x, cylinder_half_length, 0.0])
        return np.array([params.half_size_x, params.half_size_y, params.half_size_z])

    @staticmethod
    def _geom_rbound(params: ObjectParams) -> float:
        if params.shape == "cylinder":
            return float(np.hypot(params.half_size_x, params.half_size_z))
        if params.shape == "capsule":
            return params.half_size_z
        if params.shape == "ellipsoid":
            return max(params.half_size_x, params.half_size_y, params.half_size_z)
        return float(np.linalg.norm([params.half_size_x, params.half_size_y, params.half_size_z]))

    @staticmethod
    def _body_inertia(params: ObjectParams) -> tuple[float, float, float]:
        mass = params.mass
        x, y, z = params.half_size_x, params.half_size_y, params.half_size_z
        if params.shape == "box":
            return (mass * (y**2 + z**2) / 3.0, mass * (x**2 + z**2) / 3.0, mass * (x**2 + y**2) / 3.0)
        if params.shape == "ellipsoid":
            return (mass * (y**2 + z**2) / 5.0, mass * (x**2 + z**2) / 5.0, mass * (x**2 + y**2) / 5.0)
        if params.shape == "cylinder":
            transverse = mass * (3.0 * x**2 + (2.0 * z) ** 2) / 12.0
            axial = 0.5 * mass * x**2
            return (transverse, transverse, axial)

        cylinder_half_length = z - x
        cylinder_volume = np.pi * x**2 * (2.0 * cylinder_half_length)
        cap_volume = 4.0 * np.pi * x**3 / 3.0
        cylinder_mass = mass * cylinder_volume / (cylinder_volume + cap_volume)
        cap_mass = mass - cylinder_mass
        transverse = cylinder_mass * (
            3.0 * x**2 + (2.0 * cylinder_half_length) ** 2
        ) / 12.0
        transverse += cap_mass * (
            0.4 * x**2 + cylinder_half_length**2 + 0.75 * cylinder_half_length * x
        )
        axial = 0.5 * cylinder_mass * x**2 + 0.4 * cap_mass * x**2
        return (transverse, transverse, axial)

    def _reward(
        self,
        *,
        action: FloatArray,
        pad_forces: FloatArray,
        lift_height: float,
        slipped: bool,
        damaged: bool,
        dropped: bool,
        unstable: bool,
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
        elif unstable:
            reward -= 5.0
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
            "tilt_radians": self._object_tilt(),
            "control_targets": self._control_targets.copy(),
            "pad_forces": pad_forces.copy(),
            "peak_pad_force": self._peak_pad_force,
            "slip_events": self._slip_events,
            "cumulative_slip_distance": self._cumulative_slip_distance,
        }

    def _read_many(self, names: tuple[str, ...]) -> FloatArray:
        return np.array([self.data.sensordata[self._sensor_adrs[name]] for name in names], dtype=np.float32)

    def _pad_forces(self) -> FloatArray:
        return self._read_many(self.PAD_FORCE_NAMES)

    def _object_height(self) -> float:
        return float(self.data.xpos[self._object_body_id, 2])

    def _lift_height(self) -> float:
        return max(0.0, self._object_height() - self._rest_object_height)

    def _object_tilt(self) -> float:
        upright_z = float(self.data.xmat[self._object_body_id].reshape(3, 3)[2, 2])
        return float(np.arccos(np.clip(upright_z, -1.0, 1.0)))


__all__ = ["BlindTouchEnv", "EnvConfig", "ObjectParams"]
