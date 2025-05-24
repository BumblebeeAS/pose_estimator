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
            "simple_pose_estimator_node = pose_estimator.simple_pose_estimator_node:main",
            "gate_pose_estimator_node = pose_estimator.gate_pose_estimator_node:main",
        ],
    },
)
