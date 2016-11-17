# This list is required for static linking and exported to CaffeConfig.cmake
set(Caffe_LINKER_LIBS "")
set(Caffe_COMPILE_OPTIONS "")

# ---[ Boost
find_package(Boost 1.46 REQUIRED COMPONENTS system thread filesystem)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
add_definitions(-DBOOST_ALL_NO_LIB)
list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})

if(DEFINED MSVC)
  # We should define this only when necessary,
  # i.e VS 2013 Update 4 or earlier.
  add_definitions(-DBOOST_NO_CXX11_TEMPLATE_ALIASES)
endif()

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})

# ---[ Google-glog
include("cmake/External/glog.cmake")
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GLOG_LIBRARIES})

# ---[ Google-gflags
include("cmake/External/gflags.cmake")
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GFLAGS_LIBRARIES})

# ---[ Google-protobuf
include(cmake/ProtoBuf.cmake)

# ---[ HDF5
if(MSVC)
  # Find HDF5 using it's hdf5-config.cmake file with MSVC
  if(DEFINED HDF5_DIR)
    list(APPEND CMAKE_MODULE_PATH ${HDF5_DIR})
  endif()
  find_package(HDF5 COMPONENTS C HL REQUIRED)
  set(HDF5_LIBRARIES hdf5-shared)
  set(HDF5_HL_LIBRARIES hdf5_hl-shared)
else()
  find_package(HDF5 COMPONENTS HL REQUIRED)
endif()
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB REQUIRED)
  include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${LMDB_LIBRARIES})
  add_definitions(-DUSE_LMDB)
  if(ALLOW_LMDB_NOLOCK)
    add_definitions(-DALLOW_LMDB_NOLOCK)
  endif()
endif()

# ---[ LevelDB
if(USE_LEVELDB)
  find_package(LevelDB REQUIRED)
  include_directories(SYSTEM ${LevelDB_INCLUDE})
  list(APPEND Caffe_LINKER_LIBS ${LevelDB_LIBRARIES})
  add_definitions(-DUSE_LEVELDB)
endif()

# ---[ Snappy
if(USE_LEVELDB)
  find_package(Snappy REQUIRED)
  include_directories(SYSTEM ${Snappy_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${Snappy_LIBRARIES})
endif()

# ---[ CUDA
include(cmake/Cuda.cmake)
if(NOT USE_CUDA)
  message(STATUS "-- CUDA is disabled. Building without it...")
elseif(NOT HAVE_CUDA)
  set(USE_CUDA OFF)
  message(WARNING "-- CUDA is not detected by cmake. Building without it...")
endif()

# ---[ OpenCV
if(USE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  add_definitions(-DUSE_OPENCV)
endif()

# ---[ OpenMP
# Ideally, this should be provided by the BLAS library IMPORTED target. However,
# nobody does this, so we need to link to OpenMP explicitly and have the maintainer
# to flick the switch manually as needed.
if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
else()
  find_package(OpenMP QUIET)
endif()

# Moreover, OpenMP package does not provide an IMPORTED target as well, and the
# suggested way of linking to OpenMP is to append to CMAKE_{C,CXX}_FLAGS.
# However, this naïve method will force any user of Caffe to add the same kludge
# into their buildsystem again, so we put these options into per-target PUBLIC
# compile options and link flags, so that they will be exported properly.
if(OpenMP_CXX_FLAGS)
  list(APPEND Caffe_LINKER_LIBS PRIVATE ${OpenMP_CXX_FLAGS})
  list(APPEND Caffe_COMPILE_OPTIONS PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# ---[ BLAS
if(NOT APPLE)
  set(BLAS "Atlas" CACHE STRING "Selected BLAS library")
  set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

  if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
    find_package(Atlas REQUIRED)
    include_directories(SYSTEM ${Atlas_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${Atlas_LIBRARIES})
  elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
    find_package(OpenBLAS REQUIRED)
    include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${OpenBLAS_LIB})
  elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
    find_package(MKL REQUIRED)
    include_directories(SYSTEM ${MKL_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${MKL_LIBRARIES})
    add_definitions(-DUSE_MKL)
  endif()
elseif(APPLE)
  find_package(vecLib REQUIRED)
  include_directories(SYSTEM ${vecLib_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${vecLib_LINKER_LIBS})

  if(VECLIB_FOUND)
    if(NOT vecLib_INCLUDE_DIR MATCHES "^/System/Library/Frameworks/vecLib.framework.*")
      add_definitions(-DUSE_ACCELERATE)
    endif()
  endif()
endif()

# ---[ ViennaCL
if (USE_GREENTEA)
  find_package(ViennaCL)
  if (NOT ViennaCL_FOUND)
    message(FATAL_ERROR "ViennaCL required for GREENTEA but not found.")
  endif()
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  list(APPEND Caffe_LINKER_LIBS ${ViennaCL_LIBRARIES})
  add_definitions(-DUSE_GREENTEA)
  if(ViennaCL_WITH_OPENCL)
    add_definitions(-DVIENNACL_WITH_OPENCL)
  endif()

  if(USE_LIBDNN)
    add_definitions(-DUSE_LIBDNN)
  endif()

  if(USE_INTEL_SPATIAL)
    add_definitions(-DUSE_INTEL_SPATIAL)
  endif()

  if(USE_FFT)
    find_package(clFFT)
    if (NOT CLFFT_FOUND)
      message(WARNING "clFFT is not detected by cmake.Builiding without USE_FFT.")
    endif()

    find_package(fftw3)
    if (NOT FFTW3_FOUND)
      message(WARNING "fftw3 is not detected by cmake.Builiding without USE_FFT.")
    endif()

    find_package(fftw3f)
    if (NOT FFTW3F_FOUND)
      message(WARNING "fftw3f is not detected by cmake.Builiding without USE_FFT.")
    endif()

    if(CLFFT_FOUND AND FFTW3_FOUND AND FFTW3F_FOUND)
      include_directories(SYSTEM ${CLFFT_INCLUDE_DIR} ${FFTW3_INCLUDE_DIR} ${FFTW3F_INCLUDE_DIR})
      list(APPEND Caffe_LINKER_LIBS PUBLIC ${CLFFT_LIBRARY} ${FFTW3_LIBRARY} ${FFTW3F_LIBRARY})
      add_definitions(-DUSE_FFT)
    else()
      set(USE_FFT OFF)
    endif()
  endif()
endif()

if (NOT USE_GREENTEA AND NOT USE_CUDA)
  if (NOT CPU_ONLY)
    set(CPU_ONLY ON)
    message(STATUS "-- NO GPU enabled by cmake. Buildign with CPU_ONLY...")
  endif()
  add_definitions(-DCPU_ONLY)
endif()


# ---[ Python
if(BUILD_python)
  if(NOT "${python_version}" VERSION_LESS "3.0.0")
    # use python3
    find_package(PythonInterp 3.0)
    find_package(PythonLibs 3.0)
    find_package(NumPy 1.7.1)
    # Find the matching boost python implementation
    set(version ${PYTHONLIBS_VERSION_STRING})

    STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
    find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
    set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

    while(NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
      STRING( REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version} )

      STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
      find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
      set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

      STRING( REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version} )
      if("${has_more_version}" STREQUAL "")
        break()
      endif()
    endwhile()
    if(NOT Boost_PYTHON_FOUND)
      find_package(Boost 1.46 COMPONENTS python)
    endif()
  else()
    # disable Python 3 search
    find_package(PythonInterp 2.7)
    find_package(PythonLibs 2.7)
    find_package(NumPy 1.7.1)
    find_package(Boost 1.46 COMPONENTS python)
  endif()
  if(PYTHONLIBS_FOUND AND NUMPY_FOUND AND Boost_PYTHON_FOUND)
    set(HAVE_PYTHON TRUE)
    if(Boost_USE_STATIC_LIBS AND MSVC)
      add_definitions(-DBOOST_PYTHON_STATIC_LIB)
    endif()
    if(BUILD_python_layer)
      add_definitions(-DWITH_PYTHON_LAYER)
      include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
      list(APPEND Caffe_LINKER_LIBS ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
    endif()
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
