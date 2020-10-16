
###

add_executable(reprojection_error_test reprojection_error_test.cc)

target_include_directories(reprojection_error_test
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC backend_lib
)

target_link_libraries(reprojection_error_test
  PUBLIC ceres
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC backend_lib
)

