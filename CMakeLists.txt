cmake_minimum_required(VERSION 3.8.11)
project(frontend)

find_package(Boost COMPONENTS filesystem system REQUIRED)

find_package(OpenCV REQUIRED)


# brisk
find_package(brisk 2 REQUIRED)
message(STATUS "Using system brisk. Found at ${BRISK_INCLUDE_DIRS}.")

add_executable(frontend frontend.cc)

target_include_directories(frontend
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${BRISK_INCLUDE_DIRS}
)

target_link_libraries(frontend
  PUBLIC Boost::filesystem
  PUBLIC Boost::system
  PUBLIC ${OpenCV_LIBS}
  PUBLIC brisk
)
