import os
from glob import glob

from setuptools import setup

package_name = "pose_estimator"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "src"),
            glob(os.path.join("src", "*.py")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="todo",
    maintainer_email="todo@todo.com",
    description="Vision-based pose estimation",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "points_pose_estimator_node = pose_estimator.points_pose_estimator_node:main",
            "gate_pose_estimator_node = pose_estimator.gate_pose_estimator_node:main",
            "shark_fish_estimator_node = pose_estimator.shark_fish_estimator_node:main",
            "shark_fish_pose_estimator_node = pose_estimator.shark_fish_pose_estimator_node:main",
            "sos_repair_estimator_node = pose_estimator.sos_repair_estimator_node:main",
            "bin_pose_estimator_node = pose_estimator.bin_pose_estimator_node:main",
            "slalom_pose_estimator_node = pose_estimator.slalom_pose_estimator_node:main",
            "trash_detections_processor_node = pose_estimator.trash_detections_processor_node:main",
            "trash_object_in_grabber_node = pose_estimator.trash_object_in_grabber_node:main",
            "trash_pose_estimator_depth_from_odom_node = pose_estimator.trash_pose_estimator_depth_from_odom_node:main",
            "trash_pose_estimator_depth_from_table_node = pose_estimator.trash_pose_estimator_depth_from_table_node:main",
            "trash_table_pose_estimator_node = pose_estimator.trash_table_pose_estimator_node:main",
            "torpedo_pose_estimator_node = pose_estimator.torpedo_pose_estimator_node:main",
            "slalom_yaw_score_node = pose_estimator.slalom_yaw_score:main",
            "paired_objects_tracker_node = pose_estimator.paired_objects_tracker_node:main",
            "floor_pose_estimator_node = pose_estimator.floor_pose_estimator_node:main",
            "helipad_pose_estimator_node = pose_estimator.helipad_pose_estimator_node:main",
            "tins_pose_estimator_depth_from_odom_node = pose_estimator.tins_pose_estimator_depth_from_odom_node:main",
        ],
    },
)
