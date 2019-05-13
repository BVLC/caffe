# 
# All modification made by Intel Corporation: © 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

################################################################################################
# Helper function to fetch caffe includes which will be passed to dependent projects
# Usage:
#   caffe_get_current_includes(<includes_list_variable>)
function(caffe_get_current_includes includes_variable)
  get_property(current_includes DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
  caffe_convert_absolute_paths(current_includes)

  # remove at most one ${PROJECT_BINARY_DIR} include added for caffe_config.h
  list(FIND current_includes ${PROJECT_BINARY_DIR} __index)
  list(REMOVE_AT current_includes ${__index})

  # removing numpy includes (since not required for client libs)
  set(__toremove "")
  foreach(__i ${current_includes})
    if(${__i} MATCHES "python")
      list(APPEND __toremove ${__i})
    endif()
  endforeach()
  if(__toremove)
    list(REMOVE_ITEM current_includes ${__toremove})
  endif()

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
  set(install_cmake_suffix "share/Caffe")

  # ---[ Configure build-tree CaffeConfig.cmake file ]---
  caffe_get_current_includes(Caffe_INCLUDE_DIRS)

  set(Caffe_DEFINITIONS "")
  if(NOT HAVE_CUDA)
    set(HAVE_CUDA FALSE)
    list(APPEND Caffe_DEFINITIONS -DCPU_ONLY)
  endif()

  if(USE_OPENCV)
    list(APPEND Caffe_DEFINITIONS -DUSE_OPENCV)
  endif()

  if(USE_LMDB)
    list(APPEND Caffe_DEFINITIONS -DUSE_LMDB)
    if (ALLOW_LMDB_NOLOCK)
        list(APPEND Caffe_DEFINITIONS -DALLOW_LMDB_NOLOCK)
    endif()
  endif()

  if(USE_LEVELDB)
    list(APPEND Caffe_DEFINITIONS -DUSE_LEVELDB)
  endif()

  if(NOT HAVE_CUDNN)
    set(HAVE_CUDNN FALSE)
  else()
    list(APPEND DEFINITIONS -DUSE_CUDNN)
  endif()

  if(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
    list(APPEND Caffe_DEFINITIONS -DUSE_MKL)
  endif()

  configure_file("cmake/Templates/CaffeConfig.cmake.in" "${PROJECT_BINARY_DIR}/CaffeConfig.cmake" @ONLY)

  # Add targets to the build-tree export set
  export(TARGETS caffe caffeproto FILE "${PROJECT_BINARY_DIR}/CaffeTargets.cmake")
  export(PACKAGE Caffe)

  # ---[ Configure install-tree CaffeConfig.cmake file ]---

  # remove source and build dir includes
  caffe_get_items_with_prefix(${PROJECT_SOURCE_DIR} Caffe_INCLUDE_DIRS __insource)
  caffe_get_items_with_prefix(${PROJECT_BINARY_DIR} Caffe_INCLUDE_DIRS __inbinary)
  list(REMOVE_ITEM Caffe_INCLUDE_DIRS ${__insource} ${__inbinary})

  # add `install` include folder
  set(lines
     "get_filename_component(__caffe_include \"\${Caffe_CMAKE_DIR}/../../include\" ABSOLUTE)\n"
     "list(APPEND Caffe_INCLUDE_DIRS \${__caffe_include})\n"
     "unset(__caffe_include)\n")
  string(REPLACE ";" "" Caffe_INSTALL_INCLUDE_DIR_APPEND_COMMAND ${lines})

  configure_file("cmake/Templates/CaffeConfig.cmake.in" "${PROJECT_BINARY_DIR}/cmake/CaffeConfig.cmake" @ONLY)

  # Install the CaffeConfig.cmake and export set to use with install-tree
  install(FILES "${PROJECT_BINARY_DIR}/cmake/CaffeConfig.cmake" DESTINATION ${install_cmake_suffix})
  install(EXPORT CaffeTargets DESTINATION ${install_cmake_suffix})

  # ---[ Configure and install version file ]---

  # TODO: Lines below are commented because Caffe does't declare its version in headers.
  # When the declarations are added, modify `caffe_extract_caffe_version()` macro and uncomment

  # configure_file(cmake/Templates/CaffeConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/CaffeConfigVersion.cmake" @ONLY)
  # install(FILES "${PROJECT_BINARY_DIR}/CaffeConfigVersion.cmake" DESTINATION ${install_cmake_suffix})
endfunction()


