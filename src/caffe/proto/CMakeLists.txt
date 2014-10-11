project( Proto )

#	Google Protocol Buffers
find_package( Protobuf REQUIRED )

# As of Ubuntu 14.04 protoc is no longer a part of libprotobuf-dev package and should be installed
# separately as in: sudo apt-get install protobuf-compiler
if(PROTOBUF_PROTOC_EXECUTABLE)
	message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
else()
	message(FATAL_ERROR "Could not find PROTOBUF Compiler")
endif()

include_directories(${PROTOBUF_INCLUDE_DIR})
file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")
PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})

if(BUILD_PYTHON)
	include(${CMAKE_MODULE_PATH}/ProtobufPython.cmake)
    PROTOBUF_GENERATE_PYTHON(ProtoPython ${ProtoFiles})

    file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/python/caffe/proto)

    foreach(header ${ProtoPython})

        ADD_CUSTOM_COMMAND(TARGET protopython
            COMMAND cmake -E copy ${header}
            ${CMAKE_SOURCE_DIR}/python/caffe/proto/
            DEPENDS ${header}
        )

    endforeach(header)

    install(FILES ${ProtoPython} DESTINATION python/caffe/proto/)
endif()

add_library(proto
        ${ProtoSources}
        ${ProtoHeaders}
)

target_link_libraries(proto ${PROTOBUF_LIBRARIES})

# Create proto include directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/include/caffe/proto)

# Copy proto headers to include/caffe/proto/
foreach(header ${ProtoHeaders})

    ADD_CUSTOM_COMMAND(TARGET proto
        COMMAND cmake -E copy ${header}
        ${Caffe_INCLUDE_DIRS}/caffe/proto/
        DEPENDS ${header}
)

endforeach(header)
