BlindTouch ROS Notes
====================

The ROS 2 workspace lives in `ros2_ws`. The first integration target is a
MuJoCo bridge node that exposes the existing `BlindTouchEnv` over ROS topics.

Initial topics:

- `blindtouch/command`: `blindtouch_interfaces/msg/ClawCommand`
- `blindtouch/joint_states`: `sensor_msgs/msg/JointState`
- `blindtouch/tactile`: `blindtouch_interfaces/msg/TactileState`
- `blindtouch/sim_state`: `blindtouch_interfaces/msg/SimulationState`
- `blindtouch/reset`: `blindtouch_interfaces/srv/ResetSimulation`
- `blindtouch/raw_command`: unguarded `blindtouch_interfaces/msg/ClawCommand`
  from controller or policy nodes
- `blindtouch/safety_state`: `blindtouch_interfaces/msg/SafetyFilterState`

Initial nodes:

- `blindtouch_sim/blindtouch_mujoco_bridge`: sim bridge node.
- `blindtouch_control/blindtouch_scripted_controller`: scripted tactile
  controller node, implemented in C++.
- `blindtouch_control/blindtouch_safety_filter`: force/rate guard node,
  implemented in C++.
- `blindtouch_control/blindtouch_policy_node`: optional hierarchical BC policy
  node, enabled once a checkpoint path is configured. This remains Python
  because it loads the existing PyTorch policy checkpoint format.

Useful Pixi tasks:

```bash
env PIXI_HOME=$PWD/.tools/pixi .tools/pixi/bin/pixi run build-ros
env PIXI_HOME=$PWD/.tools/pixi .tools/pixi/bin/pixi run launch-sim
env PIXI_HOME=$PWD/.tools/pixi .tools/pixi/bin/pixi run launch-scripted-sim
env PIXI_HOME=$PWD/.tools/pixi .tools/pixi/bin/pixi run launch-policy-sim
```
