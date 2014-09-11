cmake_minimum_required(VERSION 2.8.8)
project( Caffe )

###    Build Options     ##########################################################################

option(CPU_ONLY "Build Caffe without GPU support" OFF)
option(BUILD_PYTHON "Build Python wrapper" OFF)
option(BUILD_MATLAB "Build Matlab wrapper" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_SHARED_LIBS "Build SHARED libs if ON and STATIC otherwise" OFF)

if(NOT BLAS)
    set(BLAS atlas)
endif()

if(NOT CUDA_TEST_DEVICE)
    set(CUDA_TEST_DEVICE -1)
endif()

#    Install Prefix
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Default install path" FORCE )
endif()

###    Configuration    ###########################################################################
#    Compiler Flags
set(CMAKE_CXX_COMPILER_FLAGS ${CMAKE_CXX_COMPILER_FLAGS} -Wall)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fPIC) # set global flags
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}) # set debug flags
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}) # set release flags

#    Global Definitions
if(CPU_ONLY)
    add_definitions(-DCPU_ONLY)
endif()

#    Include Directories
set(${PROJECT_NAME}_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
include_directories(${${PROJECT_NAME}_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)

#    CMake Scripts dir
set(CMAKE_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/CMakeScripts)

#	CMake module path for custom module finding
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SCRIPT_DIR})

#    CUDA is required globally
if(NOT CPU_ONLY)
    find_package(CUDA 5.5 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

###    Subdirectories    ##########################################################################

add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)

if(BUILD_EXAMPLES)
    message(STATUS "Examples enabled")
    add_subdirectory(examples)
endif()

if(BUILD_PYTHON)
    message(STATUS "Python enabled")
    add_subdirectory(python)
endif()

if(BUILD_MATLAB)
    message(STATUS "Matlab enabled")
    add_subdirectory(matlab)
endif()

###    Lint Target Setup    ##########################################################################

set(LINT_TARGET lint)
set(LINT_SCRIPT ${CMAKE_SCRIPT_DIR}/lint.cmake)
add_custom_target(
    ${LINT_TARGET} 
    COMMAND ${CMAKE_COMMAND} -P ${LINT_SCRIPT}
)

###    Install    #################################################################################

#    Install Includes
file(GLOB folders ${${PROJECT_NAME}_INCLUDE_DIRS}/*)
install(DIRECTORY ${folders} DESTINATION include)


