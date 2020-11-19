cmake_minimum_required(VERSION 3.8.11)
project(mini_vi_slam)

option(BUILD_TEST "build test programs" OFF)

find_package(Boost COMPONENTS filesystem system REQUIRED)

find_package(OpenCV REQUIRED)
message("-- OpenCV version: ${OpenCV_VERSION}")

find_package(Eigen3 REQUIRED)

find_package(Ceres REQUIRED)

add_subdirectory(apps)
add_subdirectory(lib)

if(BUILD_TEST)
  add_subdirectory(test)
endif(BUILD_TEST)