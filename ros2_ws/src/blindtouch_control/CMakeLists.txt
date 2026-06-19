cmake_minimum_required(VERSION 3.10)
project(blindtouch_control)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(blindtouch_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(blindtouch_scripted_controller
  src/scripted_controller_node.cpp
)
target_include_directories(blindtouch_scripted_controller PRIVATE include)
ament_target_dependencies(
  blindtouch_scripted_controller
  blindtouch_interfaces
  rclcpp
  sensor_msgs
)

add_executable(blindtouch_safety_filter
  src/safety_filter_node.cpp
)
target_include_directories(blindtouch_safety_filter PRIVATE include)
ament_target_dependencies(
  blindtouch_safety_filter
  blindtouch_interfaces
  rclcpp
  sensor_msgs
)

ament_python_install_package(${PROJECT_NAME})

install(
  TARGETS
    blindtouch_scripted_controller
    blindtouch_safety_filter
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/blindtouch_policy_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
