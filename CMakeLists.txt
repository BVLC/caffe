cmake_minimum_required(VERSION 3.9)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Caffe project
project(Caffe C CXX)

set(CMAKE_CXX_STANDARD 14)

IF (WIN32)
  SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
ENDIF()

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
IF (NOT WIN32)
    SET(CLANG-TIDY-PATH "/usr/bin/clang-tidy-7")
ENDIF()

IF (EXISTS "${CLANG-TIDY-PATH}")
  SET(CMAKE_CXX_CLANG_TIDY "${CLANG-TIDY-PATH}" "-extra-arg-before=-std=c++17" "-analyze-temporary-dtors" "-checks=*,-cppcoreguidelines-pro-bounds-array-to-pointer-decay,-cppcoreguidelines-pro-type-reinterpret-cast,-cppcoreguidelines-pro-type-vararg,-readability-implicit-bool-cast,-google-readability-braces-around-statements,-cppcoreguidelines-pro-type-union-access,-readability-braces-around-statements,-cppcoreguidelines-pro-bounds-pointer-arithmetic,-google-readability-namespace-comments,-fuchsia-default-arguments,-hicpp-no-array-decay,-llvm-namespace-comment,-readability-implicit-bool-conversion,-modernize-pass-by-value,-fuchsia-overloaded-operator,-cert-err58-cpp,-hicpp-vararg")
ENDIF()



# ---[ Caffe version
set(CAFFE_TARGET_VERSION "1.0.0" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0" CACHE STRING "Caffe soname version")
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(ExternalProject)
include(GNUInstallDirs)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

# ---[ Options
caffe_option(CPU_ONLY  "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF NOT CPU_ONLY)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -g")
endif()

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /utf-8")
endif()

caffe_set_caffe_link()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

# ---[ Warnings
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(Caffe_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})

# ---[ Includes & defines for CUDA

# cuda_compile() does not have per-call dependencies or include pathes
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected definitions and include pathes
if(HAVE_CUDA)
  # pass include pathes to cuda_include_directories()
  set(Caffe_ALL_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
  list(REMOVE_ITEM Caffe_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
  cuda_include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR} ${Caffe_ALL_INCLUDE_DIRS})

  # add definitions to nvcc flags directly
  set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
  list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)
  list(APPEND CUDA_NVCC_FLAGS ${Caffe_ALL_DEFINITIONS})
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream per-thread")
endif()

# ---[ Subdirectories
#add_subdirectory(src/gtest)
add_subdirectory(src/caffe)

# ---[ Linter target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ uninstall target
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake)

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()

