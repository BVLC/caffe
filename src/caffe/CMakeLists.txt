project( CaffeSrc )

#    Threads
find_package(Threads REQUIRED)

#	Google-glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

#   Google-gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

#   BLAS
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

#    OpenCV
find_package(OpenCV REQUIRED core highgui imgproc)
include_directories(${OpenCV_INCLUDE_DIRS})

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

#    Boost
find_package(Boost 1.46 COMPONENTS system thread REQUIRED)
include_directories( ${Boost_INCLUDE_DIR} )
link_directories( ${Boost_LIBRARY_DIRS} )

add_subdirectory(proto)

#    Recursively find source files
##    test sources
file(GLOB_RECURSE TEST_CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)

##    all cpp sources
file(GLOB_RECURSE CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

##    remove test sources from cpp sources
list(REMOVE_ITEM CPP_SOURCES ${TEST_CPP_SOURCES})

add_library(caffe ${CPP_SOURCES})
# both depend on proto
add_dependencies(caffe proto)

#    CUDA
if(NOT CPU_ONLY)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
            -gencode arch=compute_20,code=sm_20
            -gencode arch=compute_20,code=sm_21
            -gencode arch=compute_30,code=sm_30
            -gencode arch=compute_35,code=sm_35
            -gencode arch=compute_50,code=sm_50
            -gencode arch=compute_50,code=compute_50
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

    #    cuda sources
    file(GLOB_RECURSE CU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
    file(GLOB_RECURSE TEST_CU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu)
    list(REMOVE_ITEM CU_SOURCES ${TEST_CU_SOURCES})
    cuda_add_library(caffe_cu ${CU_SOURCES})
    add_dependencies(caffe_cu proto)
    target_link_libraries(caffe caffe_cu
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
    )
endif()

target_link_libraries(caffe proto
        ${BLAS_LIBRARIES}
        ${Boost_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${GFLAGS_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${HDF5_LIBRARIES}
        ${LEVELDB_LIBS}
        ${LMDB_LIBRARIES}
        ${OpenCV_LIBS}
)

#set output directory
set_target_properties(caffe PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

add_subdirectory(test)

###    Install    #################################################################################

install(TARGETS caffe DESTINATION lib)
