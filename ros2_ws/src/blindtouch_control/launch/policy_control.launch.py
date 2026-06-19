from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    policy_params = LaunchConfiguration("policy_params")
    safety_params = LaunchConfiguration("safety_params")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "policy_params",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("blindtouch_control"),
                        "config",
                        "policy_node.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "safety_params",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("blindtouch_control"),
                        "config",
                        "safety_filter.yaml",
                    ]
                ),
            ),
            Node(
                package="blindtouch_control",
                executable="blindtouch_policy_node",
                name="blindtouch_policy_node",
                output="screen",
                parameters=[policy_params],
            ),
            Node(
                package="blindtouch_control",
                executable="blindtouch_safety_filter",
                name="blindtouch_safety_filter",
                output="screen",
                parameters=[safety_params],
            ),
        ]
    )
