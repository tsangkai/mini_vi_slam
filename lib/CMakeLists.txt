add_library(backend_lib STATIC
  parameter_block.h
  sized_parameter_block.h
  landmark_parameter_block.h
  timed_3d_parameter_block.h
  timed_quat_parameter_block.h
  imu_error.h
  reprojection_error.h
)

target_include_directories(backend_lib
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(backend_lib
  PRIVATE PUBLIC ${EIGEN3_LIBS}
  PRIVATE ceres  
)