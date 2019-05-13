# generate protobuf sources
file(GLOB proto_files proto/*.proto)
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

# include python files either to force generation
add_library(caffeproto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
set(Caffe_LINKER_LIBS caffeproto ${Caffe_LINKER_LIBS}) # note, crucial to prepend!
caffe_default_properties(caffeproto)

# --[ Caffe library

# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
caffe_pickup_caffe_sources(${PROJECT_SOURCE_DIR})

if(HAVE_CUDA)
  caffe_cuda_compile(cuda_objs ${cuda})
  list(APPEND srcs ${cuda_objs} ${cuda})
endif()

add_library(caffe ${srcs})
if(MSVC)
  # Disable Boost autolinking for consuming projects
  target_compile_definitions(caffe PUBLIC -DBOOST_ALL_NO_LIB)
endif()

# Add dependency to force building MKLDNN when not given by ENV var
if(NOT DEFINED ENV{MKLDNNROOT})
add_dependencies(caffe mkldnn)
endif()
target_link_libraries(caffe caffeproto ${Caffe_LINKER_LIBS})
caffe_default_properties(caffe)
if (BUILD_SHARED_LIBS)
  set_target_properties(caffe PROPERTIES
    VERSION   ${CAFFE_TARGET_VERSION}
    SOVERSION ${CAFFE_TARGET_SOVERSION}
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    )
else()
  set_target_properties(caffe PROPERTIES
    VERSION   ${CAFFE_TARGET_VERSION}
    SOVERSION ${CAFFE_TARGET_SOVERSION}
    )
endif()

# ---[ Tests
 add_subdirectory(test)

# ---[ Install
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION include)
install(FILES ${proto_hdrs} DESTINATION include/caffe/proto)
install(TARGETS caffe caffeproto EXPORT CaffeTargets DESTINATION lib)

file(WRITE ${PROJECT_BINARY_DIR}/__init__.py)
list(APPEND proto_python ${PROJECT_BINARY_DIR}/__init__.py)
install(PROGRAMS ${proto_python} DESTINATION python/caffe/proto)


