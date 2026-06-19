# Normalized incremental command for BlindTouchEnv.step().
# Each command field is clipped to [-1.0, 1.0] by the simulator bridge.
std_msgs/Header header
float32 palm_lift
float32 finger_1_close
float32 finger_2_close
float32 finger_3_close
