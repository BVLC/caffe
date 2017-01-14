cmake_minimum_required(VERSION 2.8.7)
if(MSVC)
  # CMake 3.4 introduced a WINDOWS_EXPORT_ALL_SYMBOLS target property that makes it possible to
  # build shared libraries without using the usual declspec() decoration.
  # See: https://blog.kitware.com/create-dlls-on-windows-without-declspec-using-new-cmake-export-all-feature/
  # and https://cmake.org/cmake/help/v3.5/prop_tgt/WINDOWS_EXPORT_ALL_SYMBOLS.html
  # for details.
  cmake_minimum_required(VERSION 3.4)
endif()
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Caffe project
project(Caffe C CXX)

# ---[ Caffe version
set(CAFFE_TARGET_VERSION "1.0.0-rc3" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0-rc3" CACHE STRING "Caffe soname version")
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(ExternalProject)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)
include(cmake/WindowsCreateLinkHeader.cmake)

# ---[ Options
caffe_option(CPU_ONLY "Build Caffe without CUDA and OpenCL support" OFF)
caffe_option(USE_INDEX_64 "Build Caffe with 64 bit indexing" OFF)
caffe_option(USE_CUDA "Build Caffe with CUDA support" OFF)
caffe_option(USE_GREENTEA "Build Caffe with OpenCL support" ON)
caffe_option(USE_LIBDNN "Build Caffe with LibDNN library support" ON)
caffe_option(USE_CLBLAS "Build Caffe with clBLAS support (instead of using ViennaClBLAS)" OFF)
caffe_option(USE_CLBLAST "Build Caffe with CLBlast support (instead of using ViennaClBLAS)" OFF)
caffe_option(USE_ISAAC "Build Caffe with ISAAC support (instead of using ViennaClBLAS)" OFF)
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" OFF)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
if(MSVC)
  # default to static libs
  caffe_option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
else()
  caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
endif()
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)
caffe_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)
caffe_option(USE_FFT "Build with fftw3 or/and clFFT" OFF)

# ---[ Flag consistency check
if(CPU_ONLY)
  set(USE_CUDA OFF)
  set(USE_GREENTEA OFF)
  set(USE_CUDNN OFF)
  set(USE_LIBDNN OFF)
  set(USE_CLBLAS OFF)
  set(USE_CLBLAST OFF)
endif()

if(USE_ISAAC)
  set(USE_CLBLAS ON)
endif()

caffe_option(protobuf_MODULE_COMPATIBLE "Make the protobuf-config.cmake compatible with the module mode" ON IF MSVC)

if(MSVC AND BUILD_SHARED_LIBS)
  # Some tests (solver tests) fail when caffe is built as a shared library. The problem comes
  # from protobuf that has a global static empty_string_ variable. Since caffe and test.testbin
  # link to a static protobuf library both end up with their own instance of the empty_string_
  # variable. This causes some SEH exception to occur. In practice if the caffe executable does not link
  # to protobuf this problem should not happen. Use at your own risk.
  message(WARNING "Some tests (solvers) will fail when building as a shared library with MSVC")
endif()

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -std=c++11 -DCMAKE_BUILD")
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

caffe_set_caffe_link()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -std=c++11")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

if(USE_INDEX_64)
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_INDEX_64)
endif()

if(NOT MSVC)
  add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)
endif()

# ---[ Warnings
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${Caffe_INCLUDE_DIR} ${PROJECT_BINARY_DIR})
set(Caffe_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})
include_directories(BEFORE src) # This is needed for gtest.

# ---[ Includes & defines for CUDA

# cuda_compile() does not have per-call dependencies or include paths
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected definitions and include paths

set(Caffe_ALL_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
list(REMOVE_ITEM Caffe_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)

if(HAVE_CUDA)
  # pass include paths to cuda_include_directories()
  cuda_include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR} ${Caffe_ALL_INCLUDE_DIRS})
  # add definitions to nvcc flags directly
  list(APPEND CUDA_NVCC_FLAGS ${Caffe_ALL_DEFINITIONS})
endif()

# ---[ Subdirectories
add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)

# ---[ Linter target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ pytest target
if(BUILD_python)
  if(UNIX)
    set(python_executable python${python_version})
  else()
    set(python_executable ${PYTHON_EXECUTABLE})
  endif()
  add_custom_target(pytest COMMAND ${python_executable} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
  add_dependencies(pytest pycaffe)
endif()

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()
