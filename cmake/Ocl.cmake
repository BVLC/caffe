if(NOT USE_OCL)
  set(USE_FFT OFF)
  return()
endif()

# Detect OCL Library

function(detect_lib lib header)
  find_path(${lib}_INCLUDE_DIR NAMES ${header})
  set(${lib}_NAMES ${OCL_NAMES} ${lib})
  find_library(${lib}_LIBRARY NAMES ${${lib}_NAMES})
  include(FindPackageHandleStandardArgs)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(${lib} DEFAULT_MSG ${lib}_INCLUDE_DIR ${lib}_LIBRARY)
  if (${lib}_INCLUDE_DIR AND ${lib}_LIBRARY)
    set(HAS_${lib} TRUE PARENT_SCOPE)
  endif()
endfunction()

# Detect clBLAS
detect_lib(OpenCL CL/cl.h)
if(NOT HAS_OpenCL)
  set(USE_OCL OFF)
  message(WARNING "-- OCL is not detected by cmake. Building without USE_OCL...")
  return()
endif()

detect_lib(clBLAS clBLAS.h)
if(NOT HAS_clBLAS)
  set(USE_OCL OFF)
  message(WARNING "-- clBLAS is not detected by cmake. Building without USE_OCL...")
  return()
endif()

if(USE_FFT)
  # Detect clfft libraries.
  detect_lib(clFFT clFFT.h)
  if(NOT HAS_clFFT)
    set(USE_FFT OFF)
    message(WARNING "-- clFFT is not detected by cmake. Building without it...")
  endif()
  if(NOT (BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl"))
    detect_lib(fftw3 fftw3.h)
    detect_lib(fftw3f fftw3.h)
    if (NOT HAS_fftw3 OR NOT HAS_fftw3f)
      set(USE_FFT OFF)
      message(WARNING "-- fftw3 is not detected by cmake. Building without USE_FFT...")
    else()
      list(APPEND Caffe_LINKER_LIBS ${fftw3_LIBRARY} ${fftw3f_LIBRARY})
      include_directories(SYSTEM ${fftw3_INCLUDE_DIR})
    endif()
  endif()
endif()

if(USE_FFT)
add_definitions(-DUSE_FFT=1)
include_directories(SYSTEM ${fftw3_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${clFFT_LIBRARY})
endif()

add_definitions(-DUSE_OCL=1)
include_directories(SYSTEM ${clBLAS_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${clBLAS_LIBRARY})

include_directories(SYSTEM ${OpenCL_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${OpenCL_LIBRARY})

macro(caffe_ocl_compile_hdrs root objlist_variable)
  foreach(_hdr ${ARGN})
    string(REPLACE ".h" "_clo.o" _clo ${_hdr})
    string(REPLACE ${root} ${CMAKE_BINARY_DIR} _clo ${_clo})
    string(REPLACE "_clo.o" ".cli"  _cli ${_clo})
    add_custom_command(OUTPUT ${_clo}
      COMMAND mkdir -p `dirname ${_clo}`
      COMMAND CC=g++ CPPFLAGS="-I${root}/include" ${root}/scripts/gencl.sh ${_hdr} ${_cli}
      COMMAND ${root}/scripts/file_obj.sh ${_cli} ${_clo}
      DEPENDS ${_hdr} ${root}/scripts/gencl.sh ${root}/scripts/file_obj.sh)
      list(APPEND ${objlist_variable} ${_clo})
  endforeach()
endmacro()

macro(caffe_ocl_compile_cls root objlist_variable)
  foreach(_cl ${ARGN})
    string(REPLACE ".cl" "_clo.o" _clo ${_cl})
    string(REPLACE ${root} ${CMAKE_BINARY_DIR} _clo ${_clo})
    add_custom_command(OUTPUT ${_clo}
      COMMAND mkdir -p `dirname ${_clo}`
      COMMAND ${root}/scripts/file_obj.sh ${_cl} ${_clo}
      DEPENDS ${_cl} ${root}/scripts/file_obj.sh)
    list(APPEND ${objlist_variable} ${_clo})
  endforeach()
endmacro()
