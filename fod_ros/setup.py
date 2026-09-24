from setuptools import find_packages, setup

package_name = 'fod_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['models/best.onnx']),
        ('share/' + package_name + '/launch', ['launch/fod_detection.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Omar Dana',
    maintainer_email='omar.t.dana@hotmail.com',
    description='ROS 2 wrapper for a YOLOv8-based foreign object debris (FOD) detector',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = fod_ros.camera_node:main',
            'detector_node = fod_ros.detector_node:main',
        ],
    },
)
