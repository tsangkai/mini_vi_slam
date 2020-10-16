
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
  PUBLIC backend_lib
)

target_link_libraries(backend
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC backend_lib
)

###

add_executable(backend_direct backend_direct.cc)

target_include_directories(backend_direct
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC backend_lib
)

target_link_libraries(backend_direct
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC backend_lib
)

### 

add_executable(ground_truth_process ground_truth_process.cc)

target_include_directories(ground_truth_process
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
)

target_link_libraries(ground_truth_process
  PUBLIC ${OpenCV_LIBS}
)