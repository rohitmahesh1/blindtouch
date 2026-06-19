"""ROS node for running a learned BlindTouch hierarchical BC policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from blindtouch.hierarchical_bc import load_hierarchical_policy
from blindtouch_interfaces.msg import ClawCommand, SimulationState, TactileState

from .observation import ObservationBuilder, command_from_action


class PolicyNode(Node):
    """Load a hierarchical BC checkpoint and publish raw claw commands."""

    def __init__(self) -> None:
        super().__init__("blindtouch_policy_node")
        self._declare_parameters()
        self._builder = ObservationBuilder(
            tactile_force_scale=float(self.get_parameter("tactile_force_scale").value),
            velocity_scale=float(self.get_parameter("velocity_scale").value),
            max_episode_steps=int(self.get_parameter("max_episode_steps").value),
            history_length=int(self.get_parameter("history_length").value),
        )
        self._policy = None
        self._load_policy()

        self.create_subscription(
            TactileState,
            str(self.get_parameter("tactile_topic").value),
            self._on_tactile,
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._on_joint_state,
            10,
        )
        self.create_subscription(
            SimulationState,
            str(self.get_parameter("sim_state_topic").value),
            self._on_sim_state,
            10,
        )
        self._command_pub = self.create_publisher(
            ClawCommand,
            str(self.get_parameter("output_topic").value),
            10,
        )
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        self.create_timer(1.0 / publish_rate_hz, self._on_timer)

    def _declare_parameters(self) -> None:
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("device", "auto")
        self.declare_parameter("output_topic", "blindtouch/raw_command")
        self.declare_parameter("tactile_topic", "blindtouch/tactile")
        self.declare_parameter("joint_state_topic", "blindtouch/joint_states")
        self.declare_parameter("sim_state_topic", "blindtouch/sim_state")
        self.declare_parameter("publish_rate_hz", 25.0)
        self.declare_parameter("tactile_force_scale", 5.0)
        self.declare_parameter("velocity_scale", 0.5)
        self.declare_parameter("max_episode_steps", 120)
        self.declare_parameter("history_length", 8)
        self.declare_parameter("publish_zero_without_checkpoint", False)

    def _load_policy(self) -> None:
        checkpoint_path = str(self.get_parameter("checkpoint_path").value).strip()
        if not checkpoint_path:
            self.get_logger().warn(
                "No checkpoint_path configured; policy node will not publish learned actions"
            )
            return
        policy, metrics = load_hierarchical_policy(
            Path(checkpoint_path).expanduser(),
            device=str(self.get_parameter("device").value),
        )
        self._policy = policy
        self.get_logger().info(
            f"Loaded hierarchical BC checkpoint from {checkpoint_path}; metrics keys={list(metrics)}"
        )

    def _on_tactile(self, msg: TactileState) -> None:
        self._builder.update_tactile_state(msg)

    def _on_joint_state(self, msg: JointState) -> None:
        self._builder.update_joint_state(msg)

    def _on_sim_state(self, msg: SimulationState) -> None:
        self._builder.update_sim_state(msg)

    def _on_timer(self) -> None:
        if not self._builder.policy_ready:
            return
        if self._policy is None:
            if not bool(self.get_parameter("publish_zero_without_checkpoint").value):
                return
            action = np.zeros(4, dtype=np.float32)
        else:
            observation = self._builder.stacked_observation()
            action, _ = self._policy.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
        self._builder.set_previous_action(action)
        self._command_pub.publish(command_from_action(action, self.get_clock().now().to_msg()))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
