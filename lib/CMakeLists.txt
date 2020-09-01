add_library(backend_lib STATIC
  ParameterBlock.h
  SizedParameterBlock.h
  LandmarkParameterBlock.h
  Timed3dParameterBlock.h
  TimedQuatParameterBlock.h
  ImuError.h
)

target_include_directories(backend_lib
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(backend_lib
  PRIVATE PUBLIC ${EIGEN3_LIBS}
  PRIVATE ceres  
)