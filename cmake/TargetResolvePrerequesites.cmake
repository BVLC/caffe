set(THIS_FILE ${CMAKE_CURRENT_LIST_FILE})
set(THIS_DIR ${CMAKE_CURRENT_LIST_DIR})

include(CMakeParseArguments)

function(caffe_prerequisites_directories VAR)
  if(BUILD_SHARED_LIBS)
    # Append the caffe library output directory
    list(APPEND _directories $<TARGET_FILE_DIR:caffe>)
  endif()
  # Add boost to search directories
  list(APPEND _directories ${Boost_LIBRARY_DIRS})
  # Add gflags to search directories
  # gflags_DIR should point to root/CMake
  get_filename_component(_dir ${gflags_DIR} DIRECTORY)
  list(APPEND _directories ${_dir}/lib)
  # Add glog to search directories
  # glog_DIR should point to root/lib/cmake/glog
  get_filename_component(_dir ${glog_DIR} DIRECTORY)
  get_filename_component(_dir ${_dir} DIRECTORY)
  get_filename_component(_dir ${_dir} DIRECTORY)
  list(APPEND _directories ${_dir}/bin)
  # Add HDF5 to search directories
  # HDF5_DIR should point to root/CMake
  get_filename_component(_dir ${HDF5_DIR} DIRECTORY)
  list(APPEND _directories ${_dir}/bin)
  # Add OpenCV to search directories
  get_filename_component(_dir ${OpenCV_LIB_PATH} DIRECTORY)
  list(APPEND _directories ${_dir}/bin)
  if(CUDNN_FOUND AND HAVE_CUDNN)
    # Add OpenCV to search directories
    get_filename_component(_dir ${CUDNN_LIBRARY} DIRECTORY)
    get_filename_component(_dir ${_dir} DIRECTORY)
    get_filename_component(_dir ${_dir} DIRECTORY)
    list(APPEND _directories ${_dir}/bin)
  endif()
  if(USE_NCCL)
    # add the nvml.dll path if we are using nccl
    file(TO_CMAKE_PATH "$ENV{NVTOOLSEXT_PATH}" _nvtools_ext)
    if(NOT "${_nvtools_ext}" STREQUAL "")
      get_filename_component(_nvsmi_path ${_nvtools_ext}/../nvsmi ABSOLUTE)
      list(APPEND _directories ${_nvsmi_path})
    endif()
  endif()
  list(REMOVE_DUPLICATES _directories)
  set(${VAR} ${_directories} PARENT_SCOPE)
endfunction()

function(caffe_copy_prerequisites target)
  caffe_prerequisites_directories(_directories)
  target_copy_prerequisites(${target} ${ARGN} DIRECTORIES ${_directories})
endfunction()

function(caffe_install_prerequisites target)
  caffe_prerequisites_directories(_directories)
  target_install_prerequisites(${target} ${ARGN} DIRECTORIES ${_directories})
endfunction()

function(target_copy_prerequisites target)
  set(options USE_HARD_LINKS)
  set(oneValueArgs DESTINATION)
  set(multiValueArgs DIRECTORIES)
  cmake_parse_arguments(tcp "${options}" "${oneValueArgs}"
                            "${multiValueArgs}" ${ARGN})
  if(NOT tcp_DESTINATION)
    set(tcp_DESTINATION $<TARGET_FILE_DIR:${target}>)
  endif()
  string(REPLACE ";" "@@" tcp_DIRECTORIES "${tcp_DIRECTORIES}")
  if(USE_NCCL)
    # nccl loads the nvml.dll dynamically so we need
    # to list it explicitely
    list(APPEND _plugins nvml.dll)
  endif()
  string(REPLACE ";" "@@" _plugins "${_plugins}")
  add_custom_command(TARGET ${target} POST_BUILD
                     COMMAND ${CMAKE_COMMAND}
                             -DTARGET=$<TARGET_FILE:${target}>
                             -DDESTINATION=${tcp_DESTINATION}
                             -DUSE_HARD_LINKS=${tcp_USE_HARD_LINKS}
                             -DDIRECTORIES=${tcp_DIRECTORIES}
                             -DPLUGINS=${_plugins}
                             -P ${THIS_FILE}
                     )
endfunction()

function(target_install_prerequisites target)
  set(options )
  set(oneValueArgs DESTINATION)
  set(multiValueArgs DIRECTORIES)
  cmake_parse_arguments(tcp "${options}" "${oneValueArgs}"
                            "${multiValueArgs}" ${ARGN})
  if(NOT tcp_DESTINATION)
    set(tcp_DESTINATION bin)
  endif()
  if(NOT IS_ABSOLUTE ${tcp_DESTINATION})
    set(tcp_DESTINATION ${CMAKE_INSTALL_PREFIX}/${tcp_DESTINATION})
  endif()
  string(REPLACE ";" "@@" tcp_DIRECTORIES "${tcp_DIRECTORIES}")
  if(USE_NCCL)
    # nccl loads the nvml.dll dynamically so we need
    # to list it explicitely
    list(APPEND _plugins nvml.dll)
  endif()
  string(REPLACE ";" "@@" _plugins "${_plugins}")
  set(_command_output ${CMAKE_CURRENT_BINARY_DIR}/${target}-install-prerequisites.stamp)
  add_custom_command(OUTPUT ${_command_output}
                     COMMAND ${CMAKE_COMMAND}
                             -DTARGET=$<TARGET_FILE:${target}>
                             -DDESTINATION=${tcp_DESTINATION}
                             -DUSE_HARD_LINKS=0
                             -DDIRECTORIES=${tcp_DIRECTORIES}
                             -DPLUGINS=${_plugins}
                             -P ${THIS_FILE}
                     COMMAND ${CMAKE_COMMAND} -E touch ${_command_output}
                     )
  add_custom_target(${target}_install_prerequisites ALL
                    DEPENDS ${_command_output})
  install(FILES ${_command_output} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tmp)
endfunction()

function(create_hardlink link target result_variable)
    file(TO_NATIVE_PATH ${link} _link)
    file(TO_NATIVE_PATH ${target} _target)
    execute_process(COMMAND cmd /c mklink /H "${_link}" "${_target}"
                    RESULT_VARIABLE _result
                    OUTPUT_VARIABLE _stdout
                    ERROR_VARIABLE _stderr
                    )
    set(${result_variable} ${_result} PARENT_SCOPE)
endfunction()

function(copy_changed_file filename destination use_hard_links)
    set(_copy 1)
    set(_src_name ${filename})
    get_filename_component(_name ${_src_name} NAME)
    set(_dst_name ${destination}/${_name})

    # lock a file to ensure that no two cmake processes
    # try to copy the same file at the same time in parallel
    # builds
    string(SHA1 _hash ${_dst_name})
    set(_lock_file ${CMAKE_BINARY_DIR}/${_hash}.lock)
    file(LOCK ${_lock_file} GUARD FUNCTION)

    if(EXISTS ${_dst_name})
        file(TIMESTAMP ${_dst_name} _dst_time)
        file(TIMESTAMP ${_src_name} _src_time)
        if(${_dst_time} STREQUAL ${_src_time})
            # skip this library if the destination and source
            # have the same time stamp
            return()
        else()
            # file has changed remove
            file(REMOVE ${_dst_name})
        endif()
    endif()

    if(use_hard_links)
        message(STATUS "Creating hardlink for ${_name} in ${destination}")
        create_hardlink(${_dst_name} ${_src_name} _result)
        if(_result EQUAL 0)
            set(_copy 0)
        else()
            message(STATUS "Failed to create hardlink ${_dst_name}. Copying instead.")
        endif()
    endif()
    if(_copy)
        message(STATUS "Copying ${_name} to ${destination}")
        file(COPY ${_src_name} DESTINATION ${DESTINATION})
    endif()
endfunction()


if(CMAKE_SCRIPT_MODE_FILE)
  include(${THIS_DIR}/CaffeGetPrerequisites.cmake)
  # Recreate a list by replacing the @@ with ;
  string(REPLACE "@@" ";" DIRECTORIES "${DIRECTORIES}")
  string(REPLACE "@@" ";" PLUGINS "${PLUGINS}")
  # Get a recursive list of dependencies required by target using dumpbin
  get_prerequisites(${TARGET} _prerequisites 1 1 "" "${DIRECTORIES}")
  foreach(_prereq ${_prerequisites} ${PLUGINS})
    # Resolve the dependency using the list of directories
    gp_resolve_item("${TARGET}" "${_prereq}" "" "${DIRECTORIES}" resolved_file)
    # Copy or create hardlink (if possible)
    if(EXISTS ${resolved_file})
      copy_changed_file(${resolved_file} ${DESTINATION} ${USE_HARD_LINKS})
    endif()
  endforeach()
endif()