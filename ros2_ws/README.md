BlindTouch ROS 2 Workspace
==========================

This workspace contains ROS 2 packages for moving BlindTouch from the existing
MuJoCo/Gymnasium simulator toward ROS-driven simulation and hardware.

Current package roles:

- `blindtouch_interfaces`: custom messages and services for claw commands,
  tactile state, and simulation state.
- `blindtouch_control`: C++ scripted controller and tactile safety filter nodes,
  plus the Python learned-policy wrapper.
- `blindtouch_ros`: shared constants/helpers used by ROS-facing Python nodes.
- `blindtouch_sim`: a MuJoCo bridge node that wraps `blindtouch.env.BlindTouchEnv`.
- `blindtouch_description`: robot description scaffolding for URDF/xacro and
  MJCF assets.
- `blindtouch_hardware`: future real-claw hardware integration boundary.
- `blindtouch_bringup`: launch/config entry points for composed systems.

Before building, install the root Python package into the same environment used
by ROS:

```bash
python -m pip install -e .[test]
```

Then build from this directory in a ROS 2 environment:

```bash
colcon build --symlink-install
```

With the repo-managed Pixi/RoboStack environment, the most common commands are:

```bash
env PIXI_HOME=$PWD/.tools/pixi .tools/pixi/bin/pixi run build-ros
env PIXI_HOME=$PWD/.tools/pixi .tools/pixi/bin/pixi run launch-scripted-sim
```
