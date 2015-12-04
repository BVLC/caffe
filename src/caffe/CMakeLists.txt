<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
# generate protobuf sources
file(GLOB proto_files proto/*.proto)
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

# include python files either to force generation
add_library(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
set(Caffe_LINKER_LIBS proto ${Caffe_LINKER_LIBS}) # note, crucial to prepend!
caffe_default_properties(proto)

# --[ Caffe library

# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
caffe_pickup_caffe_sources(${PROJECT_SOURCE_DIR})

if(HAVE_CUDA)
  caffe_cuda_compile(cuda_objs ${cuda})
  list(APPEND srcs ${cuda_objs} ${cuda})
endif()

add_library(caffe ${srcs})
target_link_libraries(caffe proto ${Caffe_LINKER_LIBS})
caffe_default_properties(caffe)

# ---[ Tests
 add_subdirectory(test)

# ---[ Install
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION include)
install(FILES ${proto_hdrs} DESTINATION include/caffe/proto)
install(TARGETS caffe proto EXPORT CaffeTargets DESTINATION lib)

file(WRITE ${PROJECT_BINARY_DIR}/__init__.py)
list(APPEND proto_python ${PROJECT_BINARY_DIR}/__init__.py)
install(PROGRAMS ${proto_python} DESTINATION python/caffe/proto)


<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
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
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
