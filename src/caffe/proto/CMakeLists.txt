cmake_minimum_required(VERSION 2.8)
project( Proto )

#	Google Protocol Buffers
find_package( Protobuf REQUIRED )
include_directories(${PROTOBUF_INCLUDE_DIR})
file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")
PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})

add_library(proto
        ${ProtoSources} 
        ${ProtoHeaders}
)
        
target_link_libraries(proto ${PROTOBUF_LIBRARIES})

# Copy proto headers to include/caffe/proto/
foreach(header ${ProtoHeaders}) 

    ADD_CUSTOM_COMMAND(TARGET proto 
        COMMAND cmake -E copy ${header}
        ${Caffe_INCLUDE_DIR}/caffe/proto/
        DEPENDS ${header} 
    ) 
 
endforeach(header) 