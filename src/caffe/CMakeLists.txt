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

set(caffe_symbol_reference_header "${CMAKE_BINARY_DIR}/caffe_reference_symbols.h")
list(APPEND srcs ${caffe_symbol_reference_header})
if(MSVC)
     # convert strings to numbers
     if(HAVE_CUDNN)
         set(_have_cudnn 1)
     else()
         set(_have_cudnn 0)
     endif()
    
     if(HAVE_CUDA)
         set(_have_cuda 1)
     else()
         set(_have_cuda 0)
     endif()
    
    add_custom_command( OUTPUT ${caffe_symbol_reference_header}
                        COMMAND powershell.exe -ExecutionPolicy Bypass -Command "${CMAKE_SOURCE_DIR}/scripts/generate_symbol_reference_header.ps1 ${_have_cudnn} ${_have_cuda} | Out-File ${caffe_symbol_reference_header}"
                        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                        COMMENT "Generating header to force symbol references"
                        )        
else()
    add_custom_command( OUTPUT ${caffe_symbol_reference_header}
                        COMMAND ${CMAKE_COMMAND} -E touch ${caffe_symbol_reference_header}
    )
endif()

add_library(caffe ${srcs})
target_link_libraries(caffe proto ${Caffe_LINKER_LIBS})
caffe_default_properties(caffe)

# ---[ Tests
 add_subdirectory(test)

# ---[ Install
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION include)
install(FILES ${proto_hdrs} DESTINATION include/caffe/proto)
install(FILES ${caffe_symbol_reference_header} DESTINATION include)
install(TARGETS caffe proto EXPORT CaffeTargets DESTINATION lib)

file(WRITE ${PROJECT_BINARY_DIR}/__init__.py)
list(APPEND proto_python ${PROJECT_BINARY_DIR}/__init__.py)
install(PROGRAMS ${proto_python} DESTINATION python/caffe/proto)


