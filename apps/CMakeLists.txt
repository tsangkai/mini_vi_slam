
###

add_executable(frontend frontend.cc)

target_include_directories(frontend
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
)

target_link_libraries(frontend
  PUBLIC Boost::filesystem
  PUBLIC Boost::system
  PUBLIC ${OpenCV_LIBS}
)

###

add_executable(backend backend.cc)

target_include_directories(backend
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC lib
)

target_link_libraries(backend
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC lib
)
