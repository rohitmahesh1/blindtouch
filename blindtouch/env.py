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

from .objects import (
    EpisodeObject,
    SamplingConfig,
    episode_object_from_mapping,
    get_demo_object,
    sample_training_object,
)


FloatArray = NDArray[np.float32]


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
    contact_reward_scale: float = 0.045
    contact_balance_reward_scale: float = 0.035
    no_contact_penalty: float = 0.010
    single_contact_penalty: float = 0.018
    coordinated_probe_reward_scale: float = 0.008
    safe_force_band_reward_scale: float = 0.025
    over_force_penalty_scale: float = 0.35
    premature_lift_penalty: float = 0.025
    ready_lift_bonus: float = 0.020
    success_reward: float = 15.0
    damage_penalty: float = 8.0
    drop_penalty: float = 2.5
    unstable_penalty: float = 3.0
    timeout_penalty: float = 2.0
    no_grip_timeout_penalty: float = 1.0
    max_tilt_radians: float = np.deg2rad(40.0)
    slip_distance_threshold: float = 0.0005
    reset_clearance: float = 0.001
    maximum_settle_xy_displacement: float = 0.002
    max_reset_attempts: int = 10
    pad_center_height: float = 0.040
    pad_half_height: float = 0.027
    palm_static_sag_compensation: float = 0.008


class BlindTouchEnv(gym.Env[FloatArray, FloatArray]):
    """Touch-only mystery-object lifting environment backed by MuJoCo."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 25}

    ACTION_NAMES = ("palm_lift", "finger_1_close", "finger_2_close", "finger_3_close")
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
        sampling_config: SamplingConfig | None = None,
        render_mode: str | None = None,
        camera_name: str = "overview",
        width: int = 640,
        height: int = 480,
    ) -> None:
        super().__init__()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render mode: {render_mode!r}")

        self.config = config or EnvConfig()
        self.sampling_config = sampling_config or SamplingConfig()
        if self.config.exploration_steps >= self.config.max_episode_steps:
            raise ValueError("exploration_steps must be smaller than max_episode_steps")
        if self.config.physics_steps_per_action < 1:
            raise ValueError("physics_steps_per_action must be positive")
        if self.config.max_reset_attempts < 1:
            raise ValueError("max_reset_attempts must be positive")
        if not self.sampling_config.training_families:
            raise ValueError("training_families must contain at least one family")

        self.render_mode = render_mode
        self._width = width
        self._height = height
        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None

        asset_path = Path(xml_path) if xml_path else Path(__file__).with_name("assets") / "claw.xml"
        self.model = mujoco.MjModel.from_xml_path(str(asset_path))
        self.data = mujoco.MjData(self.model)
        try:
            self.model.camera(camera_name)
        except KeyError as error:
            raise ValueError(f"Unsupported camera: {camera_name!r}") from error
        self._camera_name = camera_name

        self._object_body_id = self.model.body("object").id
        self._object_geom_id = self.model.geom("object_geom").id
        self._compound_geom_ids = {
            name: self.model.geom(name).id
            for name in (
                "object_cabin_geom",
                "object_wheel_fl_geom",
                "object_wheel_fr_geom",
                "object_wheel_rl_geom",
                "object_wheel_rr_geom",
            )
        }
        self._accent_geom_ids = tuple(
            self.model.geom(name).id
            for name in (
                "object_accent_1_geom",
                "object_accent_2_geom",
                "object_accent_3_geom",
                "object_accent_4_geom",
                "object_accent_5_geom",
                "object_accent_6_geom",
                "object_accent_7_geom",
                "object_accent_8_geom",
                "object_accent_9_geom",
                "object_accent_10_geom",
            )
        )
        self._mutable_object_geom_ids = (
            self._object_geom_id,
            *self._compound_geom_ids.values(),
            *self._accent_geom_ids,
        )
        for geom_id in self._mutable_object_geom_ids:
            # MuJoCo optimizes XML geoms at identity as body-frame geoms. These
            # placeholders are repositioned at reset time, so keep their local
            # transforms active.
            self.model.geom_sameframe[geom_id] = int(mujoco.mjtSameFrame.mjSAMEFRAME_NONE)
        self._material_ids = {
            name: self.model.material(name).id
            for name in (
                "object",
                "orange_skin",
                "fruit_leaf",
                "tomato_skin",
                "soap_body",
                "soap_stamp",
                "car_body",
                "car_window",
                "car_tire",
                "car_lamp",
                "car_tail_lamp",
                "car_racing_stripe",
                "car_hubcap",
            )
        }
        self._pad_geom_ids = np.array(
            [self.model.geom(f"finger_{finger}_pad").id for finger in range(1, 4)],
            dtype=np.int32,
        )
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

        self.object_params: EpisodeObject | None = None
        self._step_count = 0
        self._successful_hold_steps = 0
        self._slip_events = 0
        self._peak_pad_force = 0.0
        self._rest_object_height = 0.0
        self._previous_lift_height = 0.0
        self._previous_object_xy = np.zeros(2, dtype=np.float64)
        self._rest_object_orientation = np.eye(3, dtype=np.float64)
        self._cumulative_slip_distance = 0.0
        self._previous_action = np.zeros(4, dtype=np.float32)
        self._control_targets = self._ctrl_low.copy()
        self._outcome: str | None = None
        self._initial_palm_height = 0.0
        self._initial_penetration = False
        self._settle_xy_displacement = 0.0
        self._reset_valid = True
        self._rejected_reset_samples = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Reset into an open-claw episode with a newly hidden object."""

        super().reset(seed=seed)
        episode_options = options or {}
        randomized_reset = not any(
            key in episode_options for key in ("object", "object_params", "demo_object")
        )
        self._rejected_reset_samples = 0
        for _ in range(self.config.max_reset_attempts):
            mujoco.mj_resetData(self.model, self.data)
            self.object_params = self._choose_object(episode_options)
            self._apply_object_params(self.object_params)
            self._place_object(self.object_params)
            self._control_targets = self._ctrl_low.copy()
            self._control_targets[0] = self._palm_target_for_grasp_band(self.object_params)
            self.data.qpos[self._palm_qpos_adr] = self._control_targets[0]
            self.data.ctrl[:] = self._control_targets
            mujoco.mj_forward(self.model, self.data)
            self._initial_penetration = self._has_initial_penetration()
            initial_object_xy = self.data.xpos[self._object_body_id, :2].copy()
            for _ in range(self.config.settle_steps):
                mujoco.mj_step(self.model, self.data)
            self._settle_xy_displacement = float(
                np.linalg.norm(self.data.xpos[self._object_body_id, :2] - initial_object_xy)
            )
            self._reset_valid = (
                not self._initial_penetration
                and self._settle_xy_displacement <= self.config.maximum_settle_xy_displacement
                and self._grasp_band_is_reachable(self.object_params)
            )
            if self._reset_valid or not randomized_reset:
                break
            self._rejected_reset_samples += 1
        else:
            raise RuntimeError("Unable to sample a valid object reset after maximum attempts")

        self._step_count = 0
        self._successful_hold_steps = 0
        self._slip_events = 0
        self._peak_pad_force = 0.0
        self._contact_steps = 0
        self._first_contact_step: int | None = None
        self._max_contact_count = 0
        self._rest_object_height = self._object_height()
        self._previous_lift_height = 0.0
        self._previous_object_xy = self.data.xpos[self._object_body_id, :2].copy()
        self._rest_object_orientation = self.data.xmat[self._object_body_id].reshape(3, 3).copy()
        self._cumulative_slip_distance = 0.0
        self._previous_action.fill(0.0)
        self._initial_palm_height = float(self.data.qpos[self._palm_qpos_adr])
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
        grip_metrics = self._grip_metrics(pad_forces)
        if contacting > 0:
            self._contact_steps += 1
            if self._first_contact_step is None:
                self._first_contact_step = self._step_count
        self._max_contact_count = max(self._max_contact_count, contacting)
        palm_has_lifted = (
            self.data.qpos[self._palm_qpos_adr] - self._initial_palm_height
            > self.config.attempted_lift_height
        )
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
            grip_metrics=grip_metrics,
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
            reward -= self.config.timeout_penalty
            if self._max_contact_count < 2:
                reward -= self.config.no_grip_timeout_penalty

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
            self._renderer.update_scene(self.data, camera=self._camera_name)
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

    def _choose_object(self, options: Mapping[str, Any]) -> EpisodeObject:
        provided_object = options.get("object")
        if provided_object is not None:
            if not isinstance(provided_object, EpisodeObject):
                raise TypeError("options['object'] must be an EpisodeObject")
            return provided_object
        if "demo_object" in options:
            return get_demo_object(str(options["demo_object"]), options.get("pose"))
        if "object_params" in options:
            provided_params = options["object_params"]
            if not isinstance(provided_params, Mapping):
                raise TypeError("options['object_params'] must be a mapping")
            return episode_object_from_mapping(provided_params)
        return sample_training_object(self.np_random, self.sampling_config)

    def _apply_object_params(self, params: EpisodeObject) -> None:
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
        for pad_id in self._pad_geom_ids:
            self.model.geom_friction[pad_id, :2] = params.friction
        for pair_id in self._grasp_pair_ids:
            self.model.pair_friction[pair_id, :2] = params.friction
        self._configure_compound_geometry(params)
        self._configure_visual_geometry(params)
        self.model.body_mass[self._object_body_id] = params.mass
        self.model.body_inertia[self._object_body_id] = self._body_inertia(params)

    def _place_object(self, params: EpisodeObject) -> None:
        qpos = self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 7]
        qpos[:] = (
            params.x_offset,
            params.y_offset,
            params.resting_half_height + self.config.reset_clearance,
            *params.quaternion,
        )

    def _palm_target_for_grasp_band(self, params: EpisodeObject) -> float:
        lowest_collision_clear_center = self.config.pad_half_height + self.config.reset_clearance
        requested_center = max(params.grasp_height, lowest_collision_clear_center)
        requested = (
            requested_center
            - self.config.pad_center_height
            + self.config.palm_static_sag_compensation
        )
        return float(np.clip(requested, self._ctrl_low[0], self._ctrl_high[0]))

    def _grasp_band_is_reachable(self, params: EpisodeObject) -> bool:
        pad_center = self.config.pad_center_height + self._control_targets[0]
        return bool(
            pad_center - self.config.pad_half_height
            <= params.grasp_height
            <= pad_center + self.config.pad_half_height
        )

    def _has_initial_penetration(self) -> bool:
        contact_object_ids = {self._object_geom_id, *self._compound_geom_ids.values()}
        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = {contact.geom1, contact.geom2}
            names = {self.model.geom(geom_id).name for geom_id in geom_ids}
            if geom_ids & contact_object_ids and "table" not in names:
                return True
        return False

    def _configure_compound_geometry(self, params: EpisodeObject) -> None:
        self.model.geom_pos[self._object_geom_id] = (0.0, 0.0, 0.0)
        for pad_id in self._pad_geom_ids:
            self.model.geom_conaffinity[pad_id] = 0
        for geom_id in self._compound_geom_ids.values():
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
            self.model.geom_matid[geom_id] = -1
            self.model.geom_rgba[geom_id] = (0.0, 0.0, 0.0, 0.0)
        if params.shape != "chassis":
            return

        x, y, z = params.half_size_x, params.half_size_y, params.half_size_z
        wheel_radius = min(0.0065, 0.40 * z)
        base_half_height = max(0.0045, 0.38 * z)
        base_height = -z + 2.0 * wheel_radius + base_half_height
        cabin_half_height = max(0.003, 0.22 * z)
        cabin_height = z - cabin_half_height
        wheel_x = x - 1.25 * wheel_radius
        wheel_y = y - 0.75 * wheel_radius
        wheel_height = -z + wheel_radius

        self.model.geom_size[self._object_geom_id] = (
            0.88 * x,
            0.78 * y,
            base_half_height,
        )
        self.model.geom_pos[self._object_geom_id] = (0.0, 0.0, base_height)
        self.model.geom_rbound[self._object_geom_id] = float(
            np.linalg.norm(self.model.geom_size[self._object_geom_id])
        )

        cabin_id = self._compound_geom_ids["object_cabin_geom"]
        self.model.geom_size[cabin_id] = (0.48 * x, 0.68 * y, cabin_half_height)
        self.model.geom_pos[cabin_id] = (0.0, 0.0, cabin_height)
        self.model.geom_rbound[cabin_id] = float(np.linalg.norm(self.model.geom_size[cabin_id]))
        wheel_positions = {
            "object_wheel_fl_geom": (wheel_x, wheel_y, wheel_height),
            "object_wheel_fr_geom": (wheel_x, -wheel_y, wheel_height),
            "object_wheel_rl_geom": (-wheel_x, wheel_y, wheel_height),
            "object_wheel_rr_geom": (-wheel_x, -wheel_y, wheel_height),
        }
        for name, position in wheel_positions.items():
            wheel_id = self._compound_geom_ids[name]
            self.model.geom_size[wheel_id, 0] = wheel_radius
            self.model.geom_pos[wheel_id] = position
            self.model.geom_rbound[wheel_id] = wheel_radius

        self.model.geom_rgba[cabin_id] = (0.18, 0.35, 0.78, 1.0)
        for name, geom_id in self._compound_geom_ids.items():
            self.model.geom_contype[geom_id] = 4
            self.model.geom_conaffinity[geom_id] = 1
            self.model.geom_friction[geom_id, :2] = params.friction
            if name != "object_cabin_geom":
                self.model.geom_rgba[geom_id] = (0.08, 0.08, 0.10, 1.0)
        for pad_id in self._pad_geom_ids:
            self.model.geom_conaffinity[pad_id] = 4

    def _configure_visual_geometry(self, params: EpisodeObject) -> None:
        self._show_material(self._object_geom_id, "object")
        for geom_id in self._accent_geom_ids:
            self.model.geom_matid[geom_id] = -1
            self.model.geom_rgba[geom_id] = (0.0, 0.0, 0.0, 0.0)
            self.model.geom_pos[geom_id] = (0.0, 0.0, 0.0)
            self.model.geom_quat[geom_id] = (1.0, 0.0, 0.0, 0.0)

        if params.visual_style == "orange":
            self._show_material(self._object_geom_id, "orange_skin")
            self._set_accent(
                0,
                geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=(0.002, 0.004, 0.0),
                position=(0.0, 0.0, params.half_size_z + 0.003),
                material="fruit_leaf",
            )
            self._set_accent(
                1,
                geom_type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                size=(0.006, 0.003, 0.001),
                position=(0.004, 0.0, params.half_size_z + 0.005),
                material="fruit_leaf",
            )
        elif params.visual_style == "tomato":
            self._show_material(self._object_geom_id, "tomato_skin")
            self._set_accent(
                0,
                geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=(0.0015, 0.003, 0.0),
                position=(0.0, 0.0, params.half_size_z + 0.002),
                material="fruit_leaf",
            )
            self._set_accent(
                1,
                geom_type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                size=(0.009, 0.0035, 0.001),
                position=(0.002, 0.0, params.half_size_z + 0.002),
                material="fruit_leaf",
            )
        elif params.visual_style == "soap_bar":
            self.model.geom_matid[self._object_geom_id] = -1
            self.model.geom_rgba[self._object_geom_id] = (0.0, 0.0, 0.0, 0.0)
            self._set_accent(
                0,
                geom_type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                size=(params.half_size_x, params.half_size_y, params.half_size_z),
                position=(0.0, 0.0, 0.0),
                material="soap_body",
            )
            self._set_accent(
                1,
                geom_type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                size=(0.012, 0.007, 0.0008),
                position=(0.0, 0.0, params.half_size_z + 0.0007),
                material="soap_stamp",
            )
        elif params.visual_style == "toy_car":
            self._show_material(self._object_geom_id, "car_body")
            self._show_material(self._compound_geom_ids["object_cabin_geom"], "car_window")
            for name in (
                "object_wheel_fl_geom",
                "object_wheel_fr_geom",
                "object_wheel_rl_geom",
                "object_wheel_rr_geom",
            ):
                self._show_material(self._compound_geom_ids[name], "car_tire")
            body_top_z = (
                self.model.geom_pos[self._object_geom_id, 2]
                + self.model.geom_size[self._object_geom_id, 2]
            )
            cabin_top_z = (
                self.model.geom_pos[self._compound_geom_ids["object_cabin_geom"], 2]
                + self.model.geom_size[self._compound_geom_ids["object_cabin_geom"], 2]
            )
            lamp_z = body_top_z - 0.002
            self._set_accent(
                0,
                geom_type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.0015, 0.0045, 0.0020),
                position=(params.half_size_x * 0.86, params.half_size_y * 0.42, lamp_z),
                material="car_lamp",
            )
            self._set_accent(
                1,
                geom_type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.0015, 0.0045, 0.0020),
                position=(params.half_size_x * 0.86, -params.half_size_y * 0.42, lamp_z),
                material="car_lamp",
            )
            self._set_accent(
                2,
                geom_type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.0013, 0.0038, 0.0018),
                position=(-params.half_size_x * 0.86, params.half_size_y * 0.42, lamp_z),
                material="car_tail_lamp",
            )
            self._set_accent(
                3,
                geom_type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.0013, 0.0038, 0.0018),
                position=(-params.half_size_x * 0.86, -params.half_size_y * 0.42, lamp_z),
                material="car_tail_lamp",
            )
            self._set_accent(
                4,
                geom_type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.026, 0.0012, 0.0030),
                position=(
                    0.0,
                    self.model.geom_size[self._object_geom_id, 1] + 0.0012,
                    self.model.geom_pos[self._object_geom_id, 2] + 0.0010,
                ),
                material="car_racing_stripe",
            )
            self._set_accent(
                5,
                geom_type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.026, 0.0012, 0.0030),
                position=(
                    0.0,
                    -self.model.geom_size[self._object_geom_id, 1] - 0.0012,
                    self.model.geom_pos[self._object_geom_id, 2] + 0.0010,
                ),
                material="car_racing_stripe",
            )
            for accent_index, wheel_name in enumerate(
                (
                    "object_wheel_fl_geom",
                    "object_wheel_fr_geom",
                    "object_wheel_rl_geom",
                    "object_wheel_rr_geom",
                ),
                start=6,
            ):
                wheel_id = self._compound_geom_ids[wheel_name]
                position = self.model.geom_pos[wheel_id].copy()
                side = 1.0 if position[1] >= 0.0 else -1.0
                wheel_radius = float(self.model.geom_size[wheel_id, 0])
                position[1] += side * wheel_radius * 0.80
                self._set_accent(
                    accent_index,
                    geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=(wheel_radius * 0.55, 0.0, 0.0),
                    position=tuple(float(value) for value in position),
                    material="car_hubcap",
                )

    def _show_material(self, geom_id: int, material: str) -> None:
        material_id = self._material_ids[material]
        self.model.geom_matid[geom_id] = material_id
        self.model.geom_rgba[geom_id] = self.model.mat_rgba[material_id]

    def _set_accent(
        self,
        index: int,
        *,
        geom_type: mujoco.mjtGeom,
        size: tuple[float, float, float],
        position: tuple[float, float, float],
        material: str,
    ) -> None:
        geom_id = self._accent_geom_ids[index]
        self.model.geom_type[geom_id] = int(geom_type)
        self.model.geom_size[geom_id] = size
        self.model.geom_pos[geom_id] = position
        self.model.geom_rbound[geom_id] = float(np.linalg.norm(size))
        self._show_material(geom_id, material)

    @staticmethod
    def _geom_type(shape: str) -> int:
        return {
            "cylinder": int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            "box": int(mujoco.mjtGeom.mjGEOM_BOX),
            "capsule": int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            "ellipsoid": int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
            "chassis": int(mujoco.mjtGeom.mjGEOM_BOX),
        }[shape]

    @staticmethod
    def _geom_size(params: EpisodeObject) -> NDArray[np.float64]:
        if params.shape == "cylinder":
            return np.array([params.half_size_x, params.half_size_z, 0.0])
        if params.shape == "capsule":
            cylinder_half_length = params.half_size_z - params.half_size_x
            return np.array([params.half_size_x, cylinder_half_length, 0.0])
        if params.shape == "chassis":
            return np.array(
                [0.88 * params.half_size_x, 0.78 * params.half_size_y, 0.38 * params.half_size_z]
            )
        return np.array([params.half_size_x, params.half_size_y, params.half_size_z])

    @staticmethod
    def _geom_rbound(params: EpisodeObject) -> float:
        if params.shape == "cylinder":
            return float(np.hypot(params.half_size_x, params.half_size_z))
        if params.shape == "capsule":
            return params.half_size_z
        if params.shape == "ellipsoid":
            return max(params.half_size_x, params.half_size_y, params.half_size_z)
        if params.shape == "chassis":
            return float(np.linalg.norm([params.half_size_x, params.half_size_y, params.half_size_z]))
        return float(np.linalg.norm([params.half_size_x, params.half_size_y, params.half_size_z]))

    @staticmethod
    def _body_inertia(params: EpisodeObject) -> tuple[float, float, float]:
        mass = params.mass
        x, y, z = params.half_size_x, params.half_size_y, params.half_size_z
        if params.shape in {"box", "chassis"}:
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
        grip_metrics: Mapping[str, float],
    ) -> float:
        safe_force = self.object_params.safe_force if self.object_params else 1.0
        force_fraction = float(np.max(pad_forces) / safe_force)
        contact_count = int(grip_metrics["contact_count"])
        grip_score = float(grip_metrics["grip_score"])
        balance_score = float(grip_metrics["balance_score"])
        reward = -0.004
        reward -= 0.001 * float(np.mean(np.square(action)))
        reward -= 0.003 * force_fraction**2
        if force_fraction > 0.70:
            reward -= 0.080 * (force_fraction - 0.70) ** 2

        finger_closing = np.clip(action[1:], 0.0, 1.0)
        mean_finger_close = float(np.mean(finger_closing))
        close_symmetry = 1.0 - float(np.std(finger_closing))
        if not lift_allowed and contact_count < 2:
            reward += (
                self.config.coordinated_probe_reward_scale
                * mean_finger_close
                * np.clip(close_symmetry, 0.0, 1.0)
            )
        reward += self.config.contact_reward_scale * grip_score
        if contact_count >= 2:
            reward += self.config.contact_balance_reward_scale * balance_score
            safe_margin_score = float(np.clip((0.62 - force_fraction) / 0.62, 0.0, 1.0))
            reward += self.config.safe_force_band_reward_scale * grip_score * safe_margin_score
        elif contact_count == 1:
            strongest_contact = float(np.max(pad_forces) / grip_metrics["target_pad_force"])
            reward -= self.config.single_contact_penalty * min(strongest_contact, 2.0)
        elif self._step_count > 8:
            reward -= self.config.no_contact_penalty * (1.0 - grip_score)
        if force_fraction > 0.55:
            reward -= self.config.over_force_penalty_scale * (force_fraction - 0.55) ** 2

        if lift_allowed:
            reward += 4.0 * (lift_height - self._previous_lift_height) / self.config.lift_target_height
            lift_request = max(float(action[0]), 0.0)
            if lift_request > 0.0 and grip_score < 0.45:
                reward -= self.config.premature_lift_penalty * lift_request * (1.0 - grip_score)
            elif lift_request > 0.0 and contact_count >= 2:
                reward += self.config.ready_lift_bonus * lift_request * grip_score
            if self._step_count > self.config.exploration_steps + 12 and contact_count == 0:
                reward -= self.config.no_contact_penalty
        if slipped:
            reward -= 0.25
        if damaged:
            reward -= self.config.damage_penalty
        elif unstable:
            reward -= self.config.unstable_penalty
        elif dropped:
            reward -= self.config.drop_penalty
        elif succeeded:
            reward += self.config.success_reward
        return reward

    def _grip_metrics(self, pad_forces: FloatArray) -> dict[str, float]:
        if self.object_params is None:
            return {
                "contact_count": 0.0,
                "grip_score": 0.0,
                "balance_score": 0.0,
                "target_pad_force": self.config.contact_force_threshold,
            }

        required_pad_force = self.object_params.mass * 9.81 / (
            3.0 * self.object_params.friction
        )
        target_pad_force = float(
            np.clip(
                required_pad_force * 1.15,
                self.config.contact_force_threshold * 2.0,
                self.object_params.safe_force * 0.55,
            )
        )
        normalized_forces = np.clip(pad_forces / target_pad_force, 0.0, 1.0)
        strongest_two = np.sort(normalized_forces)[-2:]
        contact_count = int(np.count_nonzero(pad_forces > self.config.contact_force_threshold))
        two_finger_score = float(np.mean(strongest_two)) if contact_count >= 2 else 0.0
        three_finger_coverage = float(np.count_nonzero(normalized_forces > 0.25) / 3.0)
        active_forces = pad_forces[pad_forces > self.config.contact_force_threshold]
        if active_forces.size >= 2:
            imbalance = float(np.std(active_forces) / max(target_pad_force, 1e-6))
            balance_score = float(np.clip(1.0 - imbalance, 0.0, 1.0))
        else:
            balance_score = 0.0
        if contact_count >= 2:
            grip_score = float(
                np.clip(0.75 * two_finger_score + 0.25 * three_finger_coverage, 0.0, 1.0)
            )
        else:
            grip_score = 0.0
        return {
            "contact_count": float(contact_count),
            "grip_score": grip_score,
            "balance_score": balance_score,
            "target_pad_force": target_pad_force,
        }

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
        grip_metrics = self._grip_metrics(pad_forces)
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
            "contact_count": int(grip_metrics["contact_count"]),
            "contact_steps": self._contact_steps,
            "first_contact_step": self._first_contact_step,
            "max_contact_count": self._max_contact_count,
            "grip_score": float(grip_metrics["grip_score"]),
            "contact_balance_score": float(grip_metrics["balance_score"]),
            "target_pad_force": float(grip_metrics["target_pad_force"]),
            "slip_events": self._slip_events,
            "cumulative_slip_distance": self._cumulative_slip_distance,
            "initial_palm_height": self._initial_palm_height,
            "initial_penetration": self._initial_penetration,
            "settle_xy_displacement": self._settle_xy_displacement,
            "reset_valid": self._reset_valid,
            "rejected_reset_samples": self._rejected_reset_samples,
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
        current_orientation = self.data.xmat[self._object_body_id].reshape(3, 3)
        relative_orientation = self._rest_object_orientation.T @ current_orientation
        cosine_angle = (float(np.trace(relative_orientation)) - 1.0) / 2.0
        return float(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


ObjectParams = EpisodeObject


class ObservationHistory(gym.Wrapper):
    """Expose recent touch/proprioception frames as one policy observation.

    The base environment remains useful for inspection and scripted baselines.
    PPO and SAC should consume this wrapper so that exploratory tactile changes
    are observable when the policy decides whether to commit to a lift.
    """

    DEFAULT_HISTORY_LENGTH = 8

    def __init__(
        self, env: BlindTouchEnv, history_length: int = DEFAULT_HISTORY_LENGTH
    ) -> None:
        super().__init__(env)
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if env.observation_space.shape != (45,):
            raise ValueError("ObservationHistory expects 45-value base observations")

        self.history_length = history_length
        self._frames = np.empty((history_length, 45), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.tile(env.observation_space.low, history_length),
            high=np.tile(env.observation_space.high, history_length),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._frames[:] = observation
        return self._stacked_observation(), info

    def step(
        self, action: FloatArray
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames[:-1] = self._frames[1:].copy()
        self._frames[-1] = observation
        return self._stacked_observation(), reward, terminated, truncated, info

    def _stacked_observation(self) -> FloatArray:
        return self._frames.reshape(-1).copy()


__all__ = [
    "BlindTouchEnv",
    "EnvConfig",
    "EpisodeObject",
    "ObjectParams",
    "ObservationHistory",
    "SamplingConfig",
]
