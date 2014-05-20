project( CaffeSrc )


#    Threads
find_package(Threads) 

#	Google-glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

#    CUDA
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
        -gencode arch=compute_20,code=sm_20
		-gencode arch=compute_20,code=sm_21
		-gencode arch=compute_30,code=sm_30
		-gencode arch=compute_35,code=sm_35
)

#    BLAS
if(BLAS STREQUAL "atlas")

    find_package(Atlas REQUIRED)
    include_directories(${ATLAS_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${ATLAS_LIBRARIES})
    
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
find_package(OpenCV COMPONENTS core highgui imgproc REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#    LevelDB
find_package(LevelDB REQUIRED)
include_directories(${LEVELDB_INCLUDE})

#    Boost
find_package(Boost 1.49 COMPONENTS system REQUIRED)
include_directories( ${Boost_INCLUDE_DIR} )
link_directories( ${Boost_LIBRARY_DIRS} )

add_subdirectory(proto)
add_subdirectory(layers)
add_subdirectory(util)

add_library(caffe
        blob.cpp
        common.cpp
        layer_factory.cpp
        net.cpp
        solver.cpp
        syncedmem.cpp
)

target_link_libraries(caffe layers proto util  

        ${GLOG_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${HDF5_LIBRARIES} 
        ${OpenCV_LIBS} 
        ${LEVELDB_LIBS} 
        ${CUDA_CUBLAS_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
)

add_subdirectory(test)