"""Names shared between BlindTouch simulation and ROS nodes."""

ACTION_NAMES = ("palm_lift", "finger_1_close", "finger_2_close", "finger_3_close")
JOINT_NAMES = ACTION_NAMES
PAD_FORCE_NAMES = tuple(f"finger_{finger}_pad_force" for finger in range(1, 4))
TAXEL_NAMES = tuple(
    f"finger_{finger}_taxel_r{row}_c{column}_force"
    for finger in range(1, 4)
    for row in range(3)
    for column in range(3)
)

BASE_OBSERVATION_SIZE = 45
HISTORY_LENGTH = 8
POLICY_OBSERVATION_SIZE = BASE_OBSERVATION_SIZE * HISTORY_LENGTH
