# This list is required for static linking and exported to CaffeConfig.cmake
set(Caffe_LINKER_LIBS "")

# ---[ Boost
find_package(Boost 1.46 REQUIRED COMPONENTS system thread OPTIONAL_COMPONENTS python)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})

# ---[ Google-glog
find_package(Glog REQUIRED)
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GLOG_LIBRARIES})

# ---[ Google-gflags
find_package(GFlags REQUIRED)
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GFLAGS_LIBRARIES})

# ---[ Google-protobuf
include(cmake/ProtoBuf.cmake)

# ---[ HDF5
find_package(HDF5 COMPONENTS HL REQUIRED)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${HDF5_LIBRARIES})

# ---[ LMDB
find_package(LMDB REQUIRED)
include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${LMDB_LIBRARIES})

# ---[ LevelDB
find_package(LevelDB REQUIRED)
include_directories(SYSTEM ${LEVELDB_INCLUDE})
list(APPEND Caffe_LINKER_LIBS ${LevelDB_LIBRARIES})

# ---[ Snappy
find_package(Snappy REQUIRED)
include_directories(SYSTEM ${Snappy_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${Snappy_LIBRARIES})

# ---[ CUDA
include(cmake/Cuda.cmake)
if(NOT HAVE_CUDA)
  if(CPU_ONLY)
    message("-- CUDA is disabled. Building without it...")
  else()
    message("-- CUDA is not detected by cmake. Building without it...")
  endif()

  # TODO: remove this not cross platform define in future. Use caffe_config.h instead.
  add_definitions(-DCPU_ONLY)
endif()

# ---[ OpenCV
find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
if(NOT OpenCV_FOUND) # temporary for OpenCV 3.x
  find_package(OpenCV REQUIRED)
endif()
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})
message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")

# ---[ BLAS
set(BLAS "Atlas" CACHE STRING "Selected BLAS library")
set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MLK")

if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")

  find_package(Atlas REQUIRED)
  include_directories(SYSTEM ${Atlas_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${Atlas_LIBRARIES})

elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")

  find_package(OpenBLAS REQUIRED)
  include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${OpenBLAS_LIB})

elseif(BLAS STREQUAL "MLK" OR BLAS STREQUAL "mkl")

  find_package(MKL REQUIRED)
  include_directories(SYSTEM ${MKL_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${MKL_LIBRARIES})

endif()

# ---[ Python
if(BUILD_python)
  # disable Python 3 search
  find_package(PythonInterp 2.7)
  find_package(PythonLibs 2.7)
  find_package(NumPy 1.7.1)
  find_package(Boost 1.46 COMPONENTS python)

  if(PYTHONLIBS_FOUND AND NUMPY_FOUND AND Boost_PYTHON_FOUND)
    set(HAVE_PYTHON TRUE)
  endif()
endif()

# ---[ Matlab
if(BUILD_matlab)
  find_package(MatlabMex)
  if(MATLABMEX_FOUND)
    set(HAVE_MATLAB TRUE)
  endif()

  # sudo apt-get install liboctave-dev
  find_program(Octave_compiler NAMES mkoctfile DOC "Octave C++ compiler")

  if(HAVE_MATLAB AND Octave_compiler)
    set(Matlab_build_mex_using "Matlab" CACHE STRING "Select Matlab or Octave if both detected")
    set_property(CACHE Matlab_build_mex_using PROPERTY STRINGS "Matlab;Octave")
  endif()
endif()

# ---[ Doxygen
if(BUILD_docs)
  find_package(Doxygen)
endif()
