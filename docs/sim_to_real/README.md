BlindTouch Sim-To-Real Notes
============================

Current priority order:

1. Keep the MuJoCo bridge and real hardware interface on the same ROS command
   and tactile-state contracts.
2. Calibrate joint zero points and actuator command ranges before policy tests.
3. Calibrate tactile taxel scale and rate limits before closed-loop lifting.
4. Run the tactile safety filter between learned policies and hardware commands.
