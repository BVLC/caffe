cmake_minimum_required(VERSION 2.8)

if(ANDROID)
  add_library(caffe_jni SHARED caffe_jni.cpp caffe_mobile.cpp)
  #add_executable(caffe_jni caffe_jni.cpp caffe_mobile.cpp)
  target_link_libraries(caffe_jni ${Caffe_LINK})
  caffe_default_properties(caffe_jni)

  # set back RUNTIME_OUTPUT_DIRECTORY
  set_target_properties(caffe_jni PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/android")

  caffe_set_solution_folder(caffe_jni android)

  # install
  install(TARGETS caffe_jni DESTINATION lib)
endif()
