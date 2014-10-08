project( CaffeSrc )


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

#    cuda sources
if(NOT CPU_ONLY)
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
        ${GFLAGS_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${HDF5_LIBRARIES}
        ${LEVELDB_LIBS}
        ${LMDB_LIBRARIES}
        ${OpenCV_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
)

#set output directory
set_target_properties(caffe PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

add_subdirectory(test)

###    Install    #################################################################################

install(TARGETS caffe DESTINATION lib)
