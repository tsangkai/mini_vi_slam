cmake_minimum_required(VERSION 3.8.11)
project(mini_vi_slam)

find_package(Boost COMPONENTS filesystem system REQUIRED)

find_package(OpenCV REQUIRED)

find_package(Eigen3 REQUIRED)

find_package(Ceres REQUIRED)



include(CTest)


add_subdirectory(apps)

add_subdirectory(lib)

add_subdirectory(test)
