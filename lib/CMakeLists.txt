add_library(backend_lib STATIC
  so3.h
  parameter_block.h
  sized_parameter_block.h
  vec_3d_parameter_block.h
  quat_parameter_block.h
  imu_error.h
  pre_int_imu_error.h
  reprojection_error.h
)

target_include_directories(backend_lib
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(backend_lib
  PRIVATE PUBLIC ${EIGEN3_LIBS}
  PRIVATE ceres  
)