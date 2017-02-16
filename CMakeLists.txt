cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Caffe project
project(Caffe C CXX)
set(CAFFE_VERSION "" CACHE STRING "Version number")

if(NOT CAFFE_VERSION)
	FIND_PROGRAM(GIT_EXECUTABLE git)
	if(GIT_EXECUTABLE)
		execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
			              WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
			              RESULT_VARIABLE VCS_BRANCH_RESULT
			              OUTPUT_VARIABLE VCS_BRANCH_VALUE
		                OUTPUT_STRIP_TRAILING_WHITESPACE
			              ERROR_VARIABLE VCS_BRANCH_ERROR
			              ERROR_QUIET)
		if ((${VCS_BRANCH_RESULT}) OR ("${VCS_BRANCH_ERROR}" NOT STREQUAL "") OR ("${VCS_BRANCH_VALUE}" STREQUAL ""))
			message(STATUS "RES: ${VCS_BRANCH_RESULT}")
			message(STATUS "VAL: ${VCS_BRANCH_VALUE}")
			message(STATUS "ERR: ${VCS_BRANCH_ERROR}")
			message(WARNING "could not get branch name from git. Command returned ${VCS_BRANCH_RESULT} with error msg ${VCS_BRANCH_ERROR}")
		else()
			set(CAFFE_VERSION "0.0.0~${VCS_BRANCH_VALUE}")
		endif()

		execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
			              WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
			              RESULT_VARIABLE VCS_HASH_RESULT
			              OUTPUT_VARIABLE VCS_HASH_VALUE
		                OUTPUT_STRIP_TRAILING_WHITESPACE
			              ERROR_VARIABLE VCS_HASH_ERROR
			              ERROR_QUIET)

		if ((${VCS_HASH_RESULT}) OR ("${VCS_HASH_ERROR}" NOT STREQUAL "") OR ("${VCS_HASH_VALUE}" STREQUAL ""))
			message(STATUS "RES: ${VCS_HASH_RESULT}")
			message(STATUS "VAL: ${VCS_HASH_VALUE}")
			message(STATUS "ERR: ${VCS_HASH_ERROR}")
			message(WARNING "could not get hash value for current version. Command returned ${VCS_HASH_RESULT} with error msg ${VCS_HASH_ERROR}")
		else()
			string(SUBSTRING ${VCS_HASH_VALUE} 0 8 VCS_HASH_VALUE)
			set(CAFFE_VERSION ${CAFFE_VERSION}~${VCS_HASH_VALUE})
		endif()

	endif(GIT_EXECUTABLE)
endif(NOT CAFFE_VERSION)
message(STATUS "Caffe version is ${CAFFE_VERSION}")

# ---[ Caffe version
set(CAFFE_TARGET_VERSION "1.0.0-rc4" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0-rc4" CACHE STRING "Caffe soname version")
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#SET(CMAKE_INSTALL_RPATH "")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_EXE_LINKER_FLAGS "-s")

include(ExternalProject)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

# ---[ Options
caffe_option(CPU_ONLY  "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF NOT CPU_ONLY)
caffe_option(USE_NCCL "Build Caffe with NCCL library support" OFF)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

caffe_set_caffe_link()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
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

# ---[ Linter target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ pytest target
if(BUILD_python)
  add_custom_target(pytest COMMAND python${python_version} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
  add_dependencies(pytest pycaffe)
endif()

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()

include(${PROJECT_SOURCE_DIR}/cmake/cpack.cmake)
