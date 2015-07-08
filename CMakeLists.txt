cmake_minimum_required(VERSION 2.8.7)

# ---[ Caffe project
project(Caffe C CXX)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

# ---[ Options
caffe_option(CPU_ONLY   "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA
caffe_option(USE_OPENCL "Build Caffe with OpenCL support" ON)
caffe_option(USE_CLGEMM "Use custom clgemm() instead of clBLASgemm()" OFF)
caffe_option(USE_OPENMP "Build Caffe with OpenMP support" OFF)
caffe_option(USE_CUDNN "Build Caffe with cuDNN libary support" OFF) # IF NOT CPU_ONLY OR USE_OPENCL)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the caffe python layer" ON)
caffe_option(ENABLE_GPROF "Build Caffe with GPROF profiling support" OFF)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

if(ENABLE_GPROF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
  if(USE_OPENCL)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
endif()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  if(USE_OPENCL)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()


add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)

# ---[ Warnings
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${Caffe_INCLUDE_DIR} ${PROJECT_BINARY_DIR})
include_directories(BEFORE src) # This is needed for gtest.

# ---[ Subdirectories
add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)
add_subdirectory(data)
add_subdirectory(src/caffe/util/OpenCL)
add_subdirectory(src/caffe/layers/OpenCL)
add_subdirectory(src/caffe/test/test_data)
add_subdirectory(examples/images)

# ---[ Linter target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()
