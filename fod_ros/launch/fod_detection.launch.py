from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='fod_ros',
            executable='camera_node',
            name='camera_node',
            output='screen',
        ),
        Node(
            package='fod_ros',
            executable='detector_node',
            name='detector_node',
            output='screen',
        ),
    ])
