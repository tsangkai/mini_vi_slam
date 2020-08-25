cmake_minimum_required(VERSION 3.8.11)
project(frontend)

find_package(Boost COMPONENTS filesystem system REQUIRED)

find_package(OpenCV REQUIRED)


# Eigen
find_package(Eigen3 REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIR})   # supposed to be global




add_executable(frontend frontend.cc)

target_include_directories(frontend
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
)

target_link_libraries(frontend
  PUBLIC Boost::filesystem
  PUBLIC Boost::system
  PUBLIC ${OpenCV_LIBS}
)

add_executable(backend backend.cc)

target_include_directories(backend
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(backend
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
)
