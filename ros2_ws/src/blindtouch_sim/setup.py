from glob import glob

from setuptools import find_packages, setup


package_name = "blindtouch_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/worlds", glob("worlds/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="BlindTouch Maintainers",
    maintainer_email="maintainers@example.com",
    description="ROS 2 simulation bridge for the BlindTouch MuJoCo environment.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "blindtouch_mujoco_bridge = blindtouch_sim.mujoco_bridge:main",
        ],
    },
)
