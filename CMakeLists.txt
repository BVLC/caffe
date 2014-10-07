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

#    Link Flags
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")  # clang++
  set(CAFFE_STATIC_LINK -Wl,-force_load $(STATIC_NAME))
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")  # g++
  set(CAFFE_STATIC_LINK -Wl,--whole-archive caffe -Wl,--no-whole-archive)
endif()

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

#	 CMake module path for custom module finding
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SCRIPT_DIR})

###    Dependencies    ##########################################################################

#    Boost
find_package(Boost 1.46 COMPONENTS system thread REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

#    CUDA
if(NOT CPU_ONLY)
    find_package(CUDA 5.5 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
            -gencode arch=compute_20,code=sm_20
            -gencode arch=compute_20,code=sm_21
            -gencode arch=compute_30,code=sm_30
            -gencode arch=compute_35,code=sm_35
    )

    # https://github.com/ComputationalRadiationPhysics/picongpu/blob/master/src/picongpu/CMakeLists.txt
    # work-arounds
    if(Boost_VERSION EQUAL 105500)
        # see https://svn.boost.org/trac/boost/ticket/9392
        message(STATUS "Boost: Applying noinline work around")
        # avoid warning for CMake >= 2.8.12
        set(CUDA_NVCC_FLAGS
          "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
    endif(Boost_VERSION EQUAL 105500)
endif()

#    Threads
find_package(Threads REQUIRED)

#	 Google-glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

#    Google-gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

#    BLAS
if(BLAS STREQUAL "atlas")

    find_package(Atlas REQUIRED)
    include_directories(${Atlas_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${Atlas_LIBRARIES})

elseif(BLAS STREQUAL "open")

    find_package(OpenBLAS REQUIRED)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${OpenBLAS_LIB})

elseif(BLAS STREQUAL "mkl")

    find_package(MKL REQUIRED)
    include_directories(${MKL_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})

endif()

#    HDF5
find_package(HDF5 COMPONENTS HL REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

#    LevelDB
find_package(LevelDB REQUIRED)
include_directories(${LEVELDB_INCLUDE})
if(LEVELDB_FOUND)
    find_package(Snappy REQUIRED)
    include_directories(${SNAPPY_INCLUDE_DIR})
    set(LEVELDB_LIBS
        ${LEVELDB_LIBS}
        ${SNAPPY_LIBS}
    )
endif()

#    LMDB
find_package(LMDB REQUIRED)
include_directories(${LMDB_INCLUDE_DIR})

#    OpenCV
find_package(OpenCV REQUIRED core highgui imgproc)
include_directories(${OpenCV_INCLUDE_DIRS})

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


