# generate protobuf sources
file(GLOB proto_files proto/*.proto)
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

# include python files either to force generation
add_library(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
caffe_default_properties(proto)
target_link_libraries(proto PUBLIC ${PROTOBUF_LIBRARIES})
target_include_directories(proto PUBLIC ${PROTOBUF_INCLUDE_DIR})

list(INSERT Caffe_LINKER_LIBS 0 PUBLIC proto) # note, crucial to prepend!

# --[ Caffe library

# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
caffe_pickup_caffe_sources(${PROJECT_SOURCE_DIR})

# add this option here since CUDA will not honor
# target_compile_definitions
if(MSVC AND NOT BUILD_SHARED_LIBS)
  set(_caffe_static_compile_def -DCAFFE_BUILDING_STATIC_LIB)
endif()

if(USE_CUDA AND HAVE_CUDA)
  caffe_cuda_compile(cuda_objs ${cuda})
  list(APPEND srcs ${cuda_objs} ${cuda})
endif()

add_library(caffe ${srcs})
target_link_libraries(caffe PUBLIC proto ${Caffe_LINKER_LIBS})
caffe_default_properties(caffe)
if(Caffe_COMPILE_OPTIONS)
  target_compile_options(caffe ${Caffe_COMPILE_OPTIONS})
endif()
set_target_properties(caffe PROPERTIES
    VERSION   ${CAFFE_TARGET_VERSION}
    SOVERSION ${CAFFE_TARGET_SOVERSION}
    )
if(MSVC AND BUILD_SHARED_LIBS)
  # CMake 3.4 introduced a WINDOWS_EXPORT_ALL_SYMBOLS target property that makes it possible to
  # build shared libraries without using the usual declspec() decoration.
  # See: https://blog.kitware.com/create-dlls-on-windows-without-declspec-using-new-cmake-export-all-feature/
  # and https://cmake.org/cmake/help/v3.5/prop_tgt/WINDOWS_EXPORT_ALL_SYMBOLS.html
  # for details.
  set_target_properties(caffe PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
elseif(MSVC AND NOT BUILD_SHARED_LIBS)
  # add a custom build command that generates a list of symbols
  # to force linking. This is required because MSVC as nothing
  # the whole-archive option
  windows_create_link_header(caffe ${caffe_symbols_hdr})
  get_filename_component(_name ${caffe_symbols_hdr} NAME)
  set(CAFFE_INCLUDE_SYMBOLS "#include \"caffe/${_name}\"")
  # definition needed to include CMake generated files
  target_compile_definitions(caffe PRIVATE ${_caffe_static_compile_def}
                                   PUBLIC -DCMAKE_WINDOWS_BUILD)
endif()
if(MSVC)
  # Disable Boost autolinking for consuming projects
  target_compile_definitions(caffe PUBLIC -DBOOST_ALL_NO_LIB)
endif()

configure_file(${caffe_export_hdr_in} ${caffe_export_hdr})

# ---[ Tests
 add_subdirectory(test)

# ---[ Install
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION include)
install(FILES ${proto_hdrs} DESTINATION include/caffe/proto)
install(TARGETS caffe proto EXPORT CaffeTargets DESTINATION lib)
if(MSVC AND NOT BUILD_SHARED_LIBS)
  install(FILES ${caffe_export_hdr} ${caffe_symbols_hdr} DESTINATION include/caffe)
endif()

file(WRITE ${PROJECT_BINARY_DIR}/__init__.py)
list(APPEND proto_python ${PROJECT_BINARY_DIR}/__init__.py)
install(PROGRAMS ${proto_python} DESTINATION python/caffe/proto)

