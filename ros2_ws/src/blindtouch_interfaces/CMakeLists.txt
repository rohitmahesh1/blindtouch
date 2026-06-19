cmake_minimum_required(VERSION 3.10)
project(blindtouch_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ClawCommand.msg"
  "msg/SafetyFilterState.msg"
  "msg/SimulationState.msg"
  "msg/TactileState.msg"
  "srv/ResetSimulation.srv"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
