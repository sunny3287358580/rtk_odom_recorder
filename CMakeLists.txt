cmake_minimum_required(VERSION 3.8)
project(rtk_odom_recorder)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(bt982k6_msgs REQUIRED)   # 改成你的包名
find_package(Eigen3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(rtk_odom_recorder src/rtk_path_recorder.cpp)
target_include_directories(rtk_odom_recorder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(rtk_odom_recorder
  rclcpp sensor_msgs nav_msgs geometry_msgs bt982k6_msgs)

install(TARGETS rtk_odom_recorder DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/align_trajectories.py
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  enable_testing()
  add_test(
    NAME align_trajectories
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_align_trajectories.py)
endif()

ament_package()
