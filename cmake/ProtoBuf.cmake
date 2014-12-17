# Finds Google Protocol Buffers library and compilers and extends
# the standart cmake script with version and python generation support

find_package( Protobuf REQUIRED )
include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${PROTOBUF_LIBRARIES})

# As of Ubuntu 14.04 protoc is no longer a part of libprotobuf-dev package
# and should be installed  separately as in: sudo apt-get install protobuf-compiler
if(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
  message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
else()
  message(FATAL_ERROR "Could not find PROTOBUF Compiler")
endif()

if(PROTOBUF_FOUND)
  # fetches protobuf version
  caffe_parse_header(${PROTOBUF_INCLUDE_DIR}/google/protobuf/stubs/common.h VERION_LINE GOOGLE_PROTOBUF_VERSION)
  string(REGEX MATCH "([0-9])00([0-9])00([0-9])" PROTOBUF_VERSION ${GOOGLE_PROTOBUF_VERSION})
  set(PROTOBUF_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
  unset(GOOGLE_PROTOBUF_VERSION)
endif()

# place where to generate protobuf sources
set(proto_gen_folder "${CMAKE_BINARY_DIR}/include/caffe/proto")
include_directories(SYSTEM "${CMAKE_BINARY_DIR}/include")

set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)

################################################################################################
# Modification of standard 'protobuf_generate_cpp()' with support of setting output directory
# Usage:
#   caffe_protobuf_generate_cpp(<output_directory> <srcs_variable> <hdrs_variable> <proto_files_list>)
function(caffe_protobuf_generate_cpp output_dir srcs_var hdrs_var)
  if(NOT ARGN)
    message(SEND_ERROR "Error: caffe_protobuf_generate_cpp() called without any proto files")
    return()
  endif()

  file(MAKE_DIRECTORY ${output_dir})

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(fil ${ARGN})
      get_filename_component(abs_fil ${fil} ABSOLUTE)
      get_filename_component(abs_path ${abs_fil} PATH)
      list(FIND _protobuf_include_path ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${abs_path})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
    foreach(dir ${PROTOBUF_IMPORT_DIRS})
      get_filename_component(abs_path ${dir} ABSOLUTE)
      list(FIND _protobuf_include_path ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${abs_path})
      endif()
    endforeach()
  endif()

  set(${srcs_var})
  set(${hdrs_var})
  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)

    list(APPEND ${srcs_var} "${output_dir}/${fil_we}.pb.cc")
    list(APPEND ${hdrs_var} "${output_dir}/${fil_we}.pb.h")

    # note, the command is executed only if output file(s) doesn't exist.
    # to execture the command every build need to convert it to PRE_BUILD cmd
    # i.e. add target name parameter to the macro and replace
    # "OUTPUT ${files}" with "TARGET ${target} PRE_BUILD"
    add_custom_command(
      OUTPUT "${output_dir}/${fil_we}.pb.cc"
             "${output_dir}/${fil_we}.pb.h"
      COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --cpp_out  ${output_dir} ${_protobuf_include_path} ${abs_fil}
      DEPENDS ${ABS_FIL}
      COMMENT "Running C++ protocol buffer compiler on ${fil}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${srcs_var}} ${${hdrs_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
  set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
endfunction()



################################################################################################
# Extention of protobuf scripts for python generatipon with output directory supoprt. This is
# not implemented in standard FindProtoBuf.cmake file
# Usage:
#   caffe_protobuf_generate_py(<output_directory> <srcs_variable> <proto_files_list>)
function(caffe_protobuf_generate_py output_dir srcs_var)
  if(NOT ARGN)
    message(SEND_ERROR "Error: caffe_protobuf_generate_python() called without any proto files")
    return()
  endif(NOT ARGN)

  file(MAKE_DIRECTORY ${output_dir})
  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(fil ${ARGN})
      get_filename_component(abs_fil ${fil} ABSOLUTE)
      get_filename_component(abs_path ${abs_fil} PATH)
      list(FIND _protobuf_include_path ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${abs_path})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
    foreach(dir ${PROTOBUF_IMPORT_DIRS})
      get_filename_component(abs_path ${dir} ABSOLUTE)
      list(FIND _protobuf_include_path ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${abs_path})
      endif()
    endforeach()
  endif()

  set(${srcs_var})
  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)

    list(APPEND ${srcs_var} "${output_dir}/${fil_we}_pb2.py")

    # note, the command is executed only if output file(s) doesn't exist.
    # to execture the command every build need to convert it to PRE_BUILD cmd
    # i.e. add target name parameter to the macro and replace
    # "OUTPUT ${files}" with "TARGET ${target} PRE_BUILD"
    add_custom_command(
      OUTPUT "${output_dir}/${fil_we}_pb2.py"
      COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --python_out ${output_dir} ${_protobuf_include_path} ${abs_fil}
      DEPENDS ${abs_fil}
      COMMENT "Running Python protocol buffer compiler on ${fil}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${srcs_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
endfunction()




