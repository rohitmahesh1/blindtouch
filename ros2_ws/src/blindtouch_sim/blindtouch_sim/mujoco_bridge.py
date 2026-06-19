"""ROS 2 node that exposes BlindTouchEnv as topics and a reset service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from blindtouch.env import BlindTouchEnv, EnvConfig
from blindtouch_interfaces.msg import ClawCommand, SimulationState, TactileState
from blindtouch_interfaces.srv import ResetSimulation
from blindtouch_ros.constants import JOINT_NAMES


class BlindTouchMujocoBridge(Node):
    """Publish MuJoCo claw state and consume normalized claw commands."""

    def __init__(self) -> None:
        super().__init__("blindtouch_mujoco_bridge")
        self._declare_parameters()

        self._frame_id = str(self.get_parameter("frame_id").value)
        self._command_topic = str(self.get_parameter("command_topic").value)
        self._joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self._tactile_topic = str(self.get_parameter("tactile_topic").value)
        self._sim_state_topic = str(self.get_parameter("sim_state_topic").value)
        self._auto_reset = bool(self.get_parameter("auto_reset").value)

        self._action = np.zeros(4, dtype=np.float32)
        self._reward = 0.0
        self._terminated = False
        self._truncated = False

        self._env = self._make_env()
        self._observation, self._info = self._reset_env_from_parameters()

        self._command_sub = self.create_subscription(
            ClawCommand,
            self._command_topic,
            self._on_command,
            10,
        )
        self._joint_pub = self.create_publisher(JointState, self._joint_state_topic, 10)
        self._tactile_pub = self.create_publisher(TactileState, self._tactile_topic, 10)
        self._sim_state_pub = self.create_publisher(SimulationState, self._sim_state_topic, 10)
        self._reset_srv = self.create_service(
            ResetSimulation,
            "blindtouch/reset",
            self._on_reset,
        )

        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._on_timer)
        self.get_logger().info(
            f"BlindTouch MuJoCo bridge publishing at {publish_rate_hz:.1f} Hz"
        )

    def _declare_parameters(self) -> None:
        default_config = EnvConfig()
        self.declare_parameter("xml_path", "")
        self.declare_parameter("camera_name", "overview")
        self.declare_parameter("frame_id", "blindtouch_world")
        self.declare_parameter("seed", 0)
        self.declare_parameter("use_seed", True)
        self.declare_parameter("publish_rate_hz", 25.0)
        self.declare_parameter("auto_reset", False)
        self.declare_parameter("physics_steps_per_action", default_config.physics_steps_per_action)
        self.declare_parameter("exploration_steps", default_config.exploration_steps)
        self.declare_parameter("max_episode_steps", default_config.max_episode_steps)
        self.declare_parameter("command_topic", "blindtouch/command")
        self.declare_parameter("joint_state_topic", "blindtouch/joint_states")
        self.declare_parameter("tactile_topic", "blindtouch/tactile")
        self.declare_parameter("sim_state_topic", "blindtouch/sim_state")

    def _make_env(self) -> BlindTouchEnv:
        xml_path_param = str(self.get_parameter("xml_path").value).strip()
        xml_path = Path(xml_path_param).expanduser() if xml_path_param else None
        config = EnvConfig(
            physics_steps_per_action=int(self.get_parameter("physics_steps_per_action").value),
            exploration_steps=int(self.get_parameter("exploration_steps").value),
            max_episode_steps=int(self.get_parameter("max_episode_steps").value),
        )
        return BlindTouchEnv(
            xml_path=xml_path,
            config=config,
            camera_name=str(self.get_parameter("camera_name").value),
        )

    def _reset_env_from_parameters(self) -> tuple[np.ndarray, dict[str, Any]]:
        if bool(self.get_parameter("use_seed").value):
            seed = int(self.get_parameter("seed").value)
        else:
            seed = None
        return self._reset_env(seed)

    def _reset_env(self, seed: int | None) -> tuple[np.ndarray, dict[str, Any]]:
        self._action[:] = 0.0
        self._reward = 0.0
        self._terminated = False
        self._truncated = False
        observation, info = self._env.reset(seed=seed)
        self.get_logger().info(f"Reset BlindTouch MuJoCo env with seed={seed}")
        return observation, info

    def _on_command(self, msg: ClawCommand) -> None:
        self._action = np.clip(
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

    def _on_reset(
        self,
        request: ResetSimulation.Request,
        response: ResetSimulation.Response,
    ) -> ResetSimulation.Response:
        seed = int(request.seed) if request.use_seed else None
        try:
            self._observation, self._info = self._reset_env(seed)
        except Exception as error:  # pragma: no cover - surfaced through ROS service response.
            response.ok = False
            response.message = str(error)
            return response
        response.ok = True
        response.message = "reset complete"
        return response

    def _on_timer(self) -> None:
        if self._terminated or self._truncated:
            if self._auto_reset:
                self._observation, self._info = self._reset_env_from_parameters()
            self._publish_state()
            return

        (
            self._observation,
            self._reward,
            self._terminated,
            self._truncated,
            self._info,
        ) = self._env.step(self._action)
        self._publish_state()

    def _publish_state(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._publish_joint_state(stamp)
        self._publish_tactile_state(stamp)
        self._publish_sim_state(stamp)

    def _publish_joint_state(self, stamp: Any) -> None:
        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.name = list(JOINT_NAMES)
        msg.position = self._read_sensor_values(self._env.JOINT_POS_NAMES)
        msg.velocity = self._read_sensor_values(self._env.JOINT_VEL_NAMES)
        msg.effort = self._read_sensor_values(self._env.EFFORT_NAMES)
        self._joint_pub.publish(msg)

    def _publish_tactile_state(self, stamp: Any) -> None:
        taxel_forces = self._read_sensor_values(self._env.TAXEL_NAMES)
        msg = TactileState()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.pad_forces = [
            float(value) for value in np.asarray(self._info["pad_forces"], dtype=np.float32)
        ]
        msg.taxel_forces = taxel_forces
        msg.max_taxel_force = float(max(taxel_forces) if taxel_forces else 0.0)
        msg.contact_count = int(self._info["contact_count"])
        self._tactile_pub.publish(msg)

    def _publish_sim_state(self, stamp: Any) -> None:
        msg = SimulationState()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.step = int(self._info["step"])
        msg.phase = str(self._info["phase"])
        msg.outcome = str(self._info["outcome"] or "")
        msg.reward = float(self._reward)
        msg.terminated = bool(self._terminated)
        msg.truncated = bool(self._truncated)
        msg.object_height = float(self._info["object_height"])
        msg.lift_height = float(self._info["lift_height"])
        msg.max_lift_height = float(self._info["max_lift_height"])
        msg.tilt_radians = float(self._info["tilt_radians"])
        msg.grip_score = float(self._info["grip_score"])
        msg.contact_balance_score = float(self._info["contact_balance_score"])
        msg.peak_pad_force = float(self._info["peak_pad_force"])
        msg.slip_events = int(self._info["slip_events"])
        msg.cumulative_slip_distance = float(self._info["cumulative_slip_distance"])
        self._sim_state_pub.publish(msg)

    def _read_sensor_values(self, names: tuple[str, ...]) -> list[float]:
        return [float(value) for value in self._env._read_many(names)]

    def destroy_node(self) -> bool:
        self._env.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = BlindTouchMujocoBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
