# generate protobuf sources
file(GLOB proto_files proto/*.proto)
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

# include python files either to force generation
add_library(caffeproto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
caffe_default_properties(caffeproto)
target_link_libraries(caffeproto PUBLIC ${PROTOBUF_LIBRARIES})
target_include_directories(caffeproto PUBLIC ${PROTOBUF_INCLUDE_DIR})

list(INSERT Caffe_LINKER_LIBS 0 PUBLIC caffeproto) # note, crucial to prepend!

# --[ Caffe library

# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
caffe_pickup_caffe_sources(${PROJECT_SOURCE_DIR})

# add this option here since CUDA will not honor
# target_compile_definitions
if(MSVC AND NOT BUILD_SHARED_LIBS)
  set(_caffe_static_compile_def -DCAFFE_BUILDING_STATIC_LIB)
endif()

if(HAVE_CUDA)
  # collect any compile definitions from imported targets. This important so that
  # preprocessor macros such as GLOG_NO_ABBREVIATED_SEVERITIES are defined.
  # this is required since CUDA macros do not honor the INTERFACE_COMPILE_DEFINITIONS
  unset(__cuda_options)
  foreach(__lib ${Caffe_LINKER_LIBS})
    if(TARGET ${__lib})
      get_target_property(__interface_compile_definitions ${__lib} INTERFACE_COMPILE_DEFINITIONS)
      if(__interface_compile_definitions)
        foreach(__def ${__interface_compile_definitions})
          # espace any parentheses because they are failing the build
          # see cmake issue https://cmake.org/Bug/view.php?id=16065
          string(REPLACE "(" "\\\(" __def_escaped ${__def})
          string(REPLACE ")" "\\\)" __def_escaped ${__def_escaped})
          # add the required -D flag
          list(APPEND __cuda_options "-D${__def_escaped}")
        endforeach()
      endif()
    endif()
  endforeach()
  list(APPEND __cuda_options ${_caffe_static_compile_def})
  # add the required definitions
  add_definitions(${__cuda_options})
  # it seems that using the OPTIONS argument like:
  # caffe_cuda_compile(cuda_objs ${cuda} OPTIONS ${__cuda_options})
  # does not work. Use add/remove_definitions instead.
  caffe_cuda_compile(cuda_objs ${cuda})
  # remove them
  remove_definitions(${__cuda_options})
  list(APPEND srcs ${cuda_objs} ${cuda})
endif()

add_library(caffe ${srcs})
caffe_default_properties(caffe)
target_link_libraries(caffe ${Caffe_LINKER_LIBS})
target_include_directories(caffe ${Caffe_INCLUDE_DIRS}
                                 PUBLIC
                                 $<BUILD_INTERFACE:${Caffe_INCLUDE_DIR}>
                                 $<INSTALL_INTERFACE:include>)
target_compile_definitions(caffe ${Caffe_DEFINITIONS})
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
if(MSVC AND USE_NCCL)
  add_dependencies(caffe nccl)
endif()

configure_file(${caffe_export_hdr_in} ${caffe_export_hdr})

# ---[ Tests
 add_subdirectory(test)

# ---[ Install
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${proto_hdrs} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/caffe/proto)
install(TARGETS caffe caffeproto EXPORT CaffeTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(MSVC AND NOT BUILD_SHARED_LIBS)
  install(FILES ${caffe_export_hdr} ${caffe_symbols_hdr} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/caffe)
endif()

file(WRITE ${PROJECT_BINARY_DIR}/__init__.py)
list(APPEND proto_python ${PROJECT_BINARY_DIR}/__init__.py)
install(PROGRAMS ${proto_python} DESTINATION python/caffe/proto)

