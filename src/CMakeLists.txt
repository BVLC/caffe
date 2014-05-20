cmake_minimum_required(VERSION 2.8)
project( Caffe )

###    Build Options     ##########################################################################

option(PYTHON "Build Python wrapper" OFF)
option(MATLAB "Build Matlab wrapper" OFF)
option(TOOLS "Build tools" ON)
option(EXAMPLES "Build examples" ON)

if(NOT BLAS)
    set(BLAS atlas)
endif()

###    Configuration    ###########################################################################

set(CMAKE_CXX_COMPILER_FLAGS ${CMAKE_CXX_COMPILER_FLAGS} -Wall -fPIC)

set(${PROJECT_NAME}_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../include)
include_directories(${${PROJECT_NAME}_INCLUDE_DIR})

#	CMake module path for custom module finding
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMakeScripts)

#    CUDA is required globally
find_package(CUDA 5.5 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


###    Subdirectories    ##########################################################################

add_subdirectory(gtest)
add_subdirectory(caffe)

if(TOOLS)
  message(STATUS "Tools enabled")
  add_subdirectory(tools)
endif()

if(EXAMPLES)
    message(STATUS "Examples enabled")
    add_subdirectory(examples)
endif()

if(PYTHON)
    message(STATUS "Python enabled")
    add_subdirectory(python)
endif()

if(MATLAB)
    message(STATUS "Matlab enabled")
    add_subdirectory(matlab)
endif()