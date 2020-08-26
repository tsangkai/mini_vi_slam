add_executable(backend_test backend_test.cc)

target_include_directories(backend_test
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC backend_lib
)

target_link_libraries(backend_test
  PUBLIC ceres
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC backend_lib
)
