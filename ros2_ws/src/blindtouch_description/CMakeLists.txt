cmake_minimum_required(VERSION 3.10)
project(blindtouch_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch meshes mjcf rviz urdf xacro
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
