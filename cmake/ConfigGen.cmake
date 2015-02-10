
################################################################################################
# Helper function to fetch caffe includes which will be passed to dependent projects
# Usage:
#   caffe_get_current_includes(<includes_list_variable>)
function(caffe_get_current_includes includes_variable)
  get_property(current_includes DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
  caffe_convert_absolute_paths(current_includes)

  # remove at most one ${CMAKE_BINARY_DIR} include added for caffe_config.h
  list(FIND current_includes ${CMAKE_BINARY_DIR} __index)
  list(REMOVE_AT current_includes ${__index})

  caffe_list_unique(current_includes)
  set(${includes_variable} ${current_includes} PARENT_SCOPE)
endfunction()

################################################################################################
# Helper function to get all list items that begin with given prefix
# Usage:
#   caffe_get_items_with_prefix(<prefix> <list_variable> <output_variable>)
function(caffe_get_items_with_prefix prefix list_variable output_variable)
  set(__result "")
  foreach(__e ${${list_variable}})
    if(__e MATCHES "^${prefix}.*")
      list(APPEND __result ${__e})
    endif()
  endforeach()
  set(${output_variable} ${__result} PARENT_SCOPE)
endfunction()

################################################################################################
# Function for generation Caffe build- and install- tree export config files
# Usage:
#  caffe_generate_export_configs()
function(caffe_generate_export_configs)
  set(install_cmake_suffix "share/caffe")

  # ---[ Configure build-tree CaffeConfig.cmake file ]---
  caffe_get_current_includes(Caffe_INCLUDE_DIRS)
  if(NOT HAVE_CUDA)
    set(HAVE_CUDA FALSE)
    set(Caffe_DEFINITIONS -DCPU_ONLY)
  endif()
  if(NOT HAVE_CUDNN)
    set(HAVE_CUDNN FALSE)
  else()
    set(Caffe_DEFINITIONS -DUSE_CUDNN)
  endif()

  configure_file("cmake/Templates/CaffeConfig.cmake.in" "${CMAKE_BINARY_DIR}/CaffeConfig.cmake" @ONLY)

  # Add targets to the build-tree export set
  export(TARGETS caffe proto FILE "${CMAKE_BINARY_DIR}/CaffeTargets.cmake")
  export(PACKAGE Caffe)

  # ---[ Configure install-tree CaffeConfig.cmake file ]---

  # remove source and build dir includes
  caffe_get_items_with_prefix(${CMAKE_SOURCE_DIR} Caffe_INCLUDE_DIRS __insource)
  caffe_get_items_with_prefix(${CMAKE_BINARY_DIR} Caffe_INCLUDE_DIRS __inbinary)
  list(REMOVE_ITEM Caffe_INCLUDE_DIRS ${__insource} ${__inbinary})

  # add `install` include folder
  set(lines "")
  list(APPEND lines "get_filename_component(__caffe_include \"\${Caffe_CMAKE_DIR}/../../include\" ABSOLUTE)\n")
  list(APPEND lines "list(APPEND Caffe_INCLUDE_DIRS \${__caffe_include})\n")
  list(APPEND lines "unset(__caffe_include)\n")
  string(REPLACE ";" "" Caffe_INSTALL_INCLUDE_DIR_APPEND_COMMAND ${lines})

  configure_file("cmake/Templates/CaffeConfig.cmake.in" "${CMAKE_BINARY_DIR}/cmake/CaffeConfig.cmake" @ONLY)

  # Install the CaffeConfig.cmake and export set to use wuth install-tree
  install(FILES "${CMAKE_BINARY_DIR}/cmake/CaffeConfig.cmake" DESTINATION ${install_cmake_suffix})
  install(EXPORT CaffeTargets DESTINATION ${install_cmake_suffix})

  # ---[ Configure and install version file ]---

  # TODO: Lines below are commented because Caffe does't declare its version in headers.
  # When the declarations are added, modify `caffe_extract_caffe_version()` macro and uncomment

  # configure_file(cmake/Templates/CaffeConfigVersion.cmake.in "${CMAKE_BINARY_DIR}/CaffeConfigVersion.cmake" @ONLY)
  # install(FILES "${CMAKE_BINARY_DIR}/CaffeConfigVersion.cmake" DESTINATION ${install_cmake_suffix})
endfunction()


