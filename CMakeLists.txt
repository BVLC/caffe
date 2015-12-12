<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> master
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Caffe project
project(Caffe C CXX)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(ExternalProject)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

# ---[ Options
caffe_option(CPU_ONLY  "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF NOT CPU_ONLY)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)

# ---[ Warnings
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${Caffe_INCLUDE_DIR} ${PROJECT_BINARY_DIR})
include_directories(BEFORE src) # This is needed for gtest.

# ---[ Subdirectories
add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)

# ---[ Linter target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ pytest target
if(BUILD_python)
  add_custom_target(pytest COMMAND python${python_version} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
  add_dependencies(pytest pycaffe)
endif()

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
# Building docs script
# Requirements:
#   sudo apt-get install doxygen texlive ruby-dev
#   sudo gem install jekyll execjs therubyracer

if(NOT BUILD_docs OR NOT DOXYGEN_FOUND)
  return()
endif()

#################################################################################################
# Gather docs from <root>/examples/**/readme.md
function(gather_readmes_as_prebuild_cmd target gathered_dir root)
  set(full_gathered_dir ${root}/${gathered_dir})

  file(GLOB_RECURSE readmes ${root}/examples/readme.md ${root}/examples/README.md)
  foreach(file ${readmes})
    # Only use file if it is to be included in docs.
    file(STRINGS ${file} file_lines REGEX "include_in_docs: true")

    if(file_lines)
      # Since everything is called readme.md, rename it by its dirname.
      file(RELATIVE_PATH file ${root} ${file})
      get_filename_component(folder ${file} PATH)
      set(new_filename ${full_gathered_dir}/${folder}.md)

      # folder value might be like <subfolder>/readme.md. That's why make directory.
      get_filename_component(new_folder ${new_filename} PATH)
      add_custom_command(TARGET ${target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${new_folder}
        COMMAND ln -sf ${root}/${file} ${new_filename}
        COMMENT "Creating symlink ${new_filename} -> ${root}/${file}"
        WORKING_DIRECTORY ${root} VERBATIM)
    endif()
  endforeach()
endfunction()

################################################################################################
# Gather docs from examples/*.ipynb and add YAML front-matter.
function(gather_notebooks_as_prebuild_cmd target gathered_dir root)
  set(full_gathered_dir ${root}/${gathered_dir})

  if(NOT PYTHON_EXECUTABLE)
    message(STATUS "Python interpeter is not found. Can't include *.ipynb files in docs. Skipping...")
    return()
  endif()

  file(GLOB_RECURSE notebooks ${root}/examples/*.ipynb)
  foreach(file ${notebooks})
    file(RELATIVE_PATH file ${root} ${file})
    set(new_filename ${full_gathered_dir}/${file})

    get_filename_component(new_folder ${new_filename} PATH)
    add_custom_command(TARGET ${target} PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${new_folder}
      COMMAND ${PYTHON_EXECUTABLE} scripts/copy_notebook.py ${file} ${new_filename}
      COMMENT "Copying notebook ${file} to ${new_filename}"
      WORKING_DIRECTORY ${root} VERBATIM)
  endforeach()

  set(${outputs_var} ${outputs} PARENT_SCOPE)
endfunction()

################################################################################################
########################## [ Non macro part ] ##################################################

# Gathering is done at each 'make doc'
file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/docs/gathered)

# Doxygen config file path
set(DOXYGEN_config_file ${PROJECT_SOURCE_DIR}/.Doxyfile CACHE FILEPATH "Doxygen config file")

# Adding docs target
add_custom_target(docs COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_config_file}
                       WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                       COMMENT "Launching doxygen..." VERBATIM)

# Gathering examples into docs subfolder
gather_notebooks_as_prebuild_cmd(docs docs/gathered ${PROJECT_SOURCE_DIR})
gather_readmes_as_prebuild_cmd(docs docs/gathered  ${PROJECT_SOURCE_DIR})

# Auto detect output directory
file(STRINGS ${DOXYGEN_config_file} config_line REGEX "OUTPUT_DIRECTORY[ \t]+=[^=].*")
if(config_line)
  string(REGEX MATCH "OUTPUT_DIRECTORY[ \t]+=([^=].*)" __ver_check "${config_line}")
  string(STRIP ${CMAKE_MATCH_1} output_dir)
  message(STATUS "Detected Doxygen OUTPUT_DIRECTORY: ${output_dir}")
else()
  set(output_dir ./doxygen/)
  message(STATUS "Can't find OUTPUT_DIRECTORY in doxygen config file. Try to use default: ${output_dir}")
endif()

if(NOT IS_ABSOLUTE ${output_dir})
  set(output_dir ${PROJECT_SOURCE_DIR}/${output_dir})
  get_filename_component(output_dir ${output_dir} ABSOLUTE)
endif()

# creates symlink in docs subfolder to code documentation built by doxygen
add_custom_command(TARGET docs POST_BUILD VERBATIM
                   COMMAND ln -sfn "${output_dir}/html" doxygen
                   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
                   COMMENT "Creating symlink ${PROJECT_SOURCE_DIR}/docs/doxygen -> ${output_dir}/html")

# for quick launch of jekyll
add_custom_target(jekyll COMMAND jekyll serve -w -s . -d _site --port=4000
                         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
                         COMMENT "Launching jekyll..." VERBATIM)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Caffe project
project(Caffe C CXX)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(ExternalProject)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

# ---[ Options
caffe_option(CPU_ONLY  "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF NOT CPU_ONLY)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)

# ---[ Warnings
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${Caffe_INCLUDE_DIR} ${PROJECT_BINARY_DIR})
include_directories(BEFORE src) # This is needed for gtest.

# ---[ Subdirectories
add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)

# ---[ Linter target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ pytest target
if(BUILD_python)
  add_custom_target(pytest COMMAND python${python_version} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
  add_dependencies(pytest pycaffe)
endif()

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()
>>>>>>> BVLC/master
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/gh-pages
=======
>>>>>>> master
=======
cmake_minimum_required(VERSION 2.8.8)
project( Caffe )

###    Build Options     ##########################################################################

option(CPU_ONLY "Build Caffe without GPU support" OFF)
option(BUILD_PYTHON "Build Python wrapper" OFF)
option(BUILD_MATLAB "Build Matlab wrapper" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_SHARED_LIBS "Build SHARED libs if ON and STATIC otherwise" OFF)

if(NOT BLAS)
    set(BLAS atlas)
endif()

if(NOT CUDA_TEST_DEVICE)
    set(CUDA_TEST_DEVICE -1)
endif()

#    Install Prefix
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Default install path" FORCE )
endif()

###    Configuration    ###########################################################################
#    Compiler Flags
set(CMAKE_CXX_COMPILER_FLAGS ${CMAKE_CXX_COMPILER_FLAGS} -Wall)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fPIC) # set global flags
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}) # set debug flags
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}) # set release flags

#    Link Flags
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")  # clang++
  set(CAFFE_STATIC_LINK -Wl,-force_load $(STATIC_NAME))
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")  # g++
  set(CAFFE_STATIC_LINK -Wl,--whole-archive caffe -Wl,--no-whole-archive)
endif()

#    Global Definitions
if(CPU_ONLY)
    add_definitions(-DCPU_ONLY)
endif()

#    Include Directories
set(${PROJECT_NAME}_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
include_directories(${${PROJECT_NAME}_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)

#    CMake Scripts dir
set(CMAKE_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/CMakeScripts)

#	 CMake module path for custom module finding
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SCRIPT_DIR})

###    Dependencies    ##########################################################################

#    Boost
find_package(Boost 1.46 COMPONENTS system thread REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

#    CUDA
if(NOT CPU_ONLY)
    find_package(CUDA 5.5 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
            -gencode arch=compute_20,code=sm_20
            -gencode arch=compute_20,code=sm_21
            -gencode arch=compute_30,code=sm_30
            -gencode arch=compute_35,code=sm_35
    )

    # https://github.com/ComputationalRadiationPhysics/picongpu/blob/master/src/picongpu/CMakeLists.txt
    # work-arounds
    if(Boost_VERSION EQUAL 105500)
        # see https://svn.boost.org/trac/boost/ticket/9392
        message(STATUS "Boost: Applying noinline work around")
        # avoid warning for CMake >= 2.8.12
        set(CUDA_NVCC_FLAGS
          "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
    endif(Boost_VERSION EQUAL 105500)
endif()

#    Threads
find_package(Threads REQUIRED)

#	 Google-glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

#    Google-gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

#    BLAS
if(BLAS STREQUAL "atlas")

    find_package(Atlas REQUIRED)
    include_directories(${Atlas_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${Atlas_LIBRARIES})

elseif(BLAS STREQUAL "open")

    find_package(OpenBLAS REQUIRED)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${OpenBLAS_LIB})

elseif(BLAS STREQUAL "mkl")

    find_package(MKL REQUIRED)
    include_directories(${MKL_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})

endif()

#    HDF5
find_package(HDF5 COMPONENTS HL REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

#    LevelDB
find_package(LevelDB REQUIRED)
include_directories(${LEVELDB_INCLUDE})
if(LEVELDB_FOUND)
    find_package(Snappy REQUIRED)
    include_directories(${SNAPPY_INCLUDE_DIR})
    set(LEVELDB_LIBS
        ${LEVELDB_LIBS}
        ${SNAPPY_LIBS}
    )
endif()

#    LMDB
find_package(LMDB REQUIRED)
include_directories(${LMDB_INCLUDE_DIR})

#    OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

###    Subdirectories    ##########################################################################

add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)

if(BUILD_EXAMPLES)
    message(STATUS "Examples enabled")
    add_subdirectory(examples)
endif()

if(BUILD_PYTHON)
    message(STATUS "Python enabled")
    add_subdirectory(python)
endif()

if(BUILD_MATLAB)
    message(STATUS "Matlab enabled")
    add_subdirectory(matlab)
endif()

###    Lint Target Setup    ##########################################################################

set(LINT_TARGET lint)
set(LINT_SCRIPT ${CMAKE_SCRIPT_DIR}/lint.cmake)
add_custom_target(
    ${LINT_TARGET}
    COMMAND ${CMAKE_COMMAND} -P ${LINT_SCRIPT}
)

###    Install    #################################################################################

#    Install Includes
file(GLOB folders ${${PROJECT_NAME}_INCLUDE_DIRS}/*)
install(DIRECTORY ${folders} DESTINATION include)


>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
