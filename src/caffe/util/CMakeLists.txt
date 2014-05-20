project( Util )

set(Util_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/util PARENT_SCOPE)
 
cuda_add_library(cuda_util
        im2col.cu
        math_functions.cu
)

add_library(util
        benchmark.cpp
        im2col.cpp
        insert_splits.cpp
        io.cpp
        math_functions.cpp
        upgrade_proto.cpp
)

target_link_libraries(util cuda_util)