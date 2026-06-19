from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("blindtouch_sim"),
                        "config",
                        "mujoco_bridge.yaml",
                    ]
                ),
            ),
            Node(
                package="blindtouch_sim",
                executable="blindtouch_mujoco_bridge",
                name="blindtouch_mujoco_bridge",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
