# This list is required for static linking and exported to CaffeConfig.cmake
set(Caffe_LINKER_LIBS "")

# ---[ Boost
find_package(Boost 1.46 REQUIRED COMPONENTS system thread filesystem regex)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})

# ---[ OpenMP
if(USE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
    set(USE_OPENMP "OFF")   # compiler is not supporting OpenMP then do not use it
  endif()
endif()

# ---[ PERFORMANCE_MONITORING
if(PERFORMANCE_MONITORING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPERFORMANCE_MONITORING")
endif()

# ---[ CPU DATA READONLY CO SIM
if(CO_SIM)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCO_SIM")
endif()

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
find_package(HDF5 COMPONENTS HL REQUIRED)
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
if(NOT HAVE_CUDA)
  if(CPU_ONLY)
    message(STATUS "-- CUDA is disabled. Building without it...")
  else()
    message(WARNING "-- CUDA is not detected by cmake. Building without it...")
  endif()

  # TODO: remove this not cross platform define in future. Use caffe_config.h instead.
  add_definitions(-DCPU_ONLY)
endif()

# ---[ OpenCV
if(USE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  add_definitions(-DUSE_OPENCV)
endif()

if(CAFFE_PER_LAYER_TIMINGS)
  add_definitions("-DCAFFE_PER_LAYER_TIMINGS")
endif()

# ---[ MLSL
if(USE_MLSL)
  if (NOT CPU_ONLY)
    message(FATAL_ERROR "Multi-node is not supported if CPU_ONLY is disabled. Please set CPU_ONLY=1 if USE_MLSL=1.")
  endif()

  #--find mlsl in external/mkl
  set(script_cmd "./external/mlsl/prepare_mlsl.sh" )
  execute_process(COMMAND ${script_cmd}
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	RESULT_VARIABLE script_result
	OUTPUT_VARIABLE RETURN_STRING)
  separate_arguments(RETURN_STRING)
  list(GET RETURN_STRING 0 MLSL_ROOT_DIR)
  list(GET RETURN_STRING 1 MLSL_LIBRARIES)
  set(MLSL_ROOT "${MLSL_ROOT_DIR}")
  #set(MLSL_ROOT "$ENV{MLSL_ROOT}")
  if(NOT MLSL_ROOT)
    message(FATAL_ERROR "Unable to find MLSL package installation directory!")
  endif()
  message(STATUS "Machine Learning Scaling Library (MLSL) found (${MLSL_ROOT}/intel64)")
  add_definitions("-DUSE_MLSL=1")
  add_definitions("-DFOUNDED_MLSL_ROOT=${MLSL_ROOT}")
  include_directories(SYSTEM "${MLSL_ROOT}/intel64/include")
  link_directories(SYSTEM "${MLSL_ROOT}/intel64/lib")
  list(APPEND Caffe_LINKER_LIBS mlsl)

  if(CAFFE_MLSL_SHUFFLE)
    add_definitions("-DCAFFE_MLSL_SHUFFLE")
  endif()
  if(FW_OVERLAP_OPT OR NOT DEFINED FW_OVERLAP_OPT)
    message(STATUS "Forward overlapping optimization is enabled!")
    add_definitions("-DFW_OVERLAP_OPT")
  endif()
endif()

# ---| DLCP
if(USE_MLSL)
  if (ENABLE_WEIGHT_GRAD_COMPRESSION OR NOT DEFINED ENABLE_WEIGHT_GRAD_COMPRESSION)
    if(DEFINED ENV{DLCPROOT})
      set(DLCPROOT_DIR $ENV{DLCPROOT})
      if(NOT ${DLCPROOT_DIR} STREQUAL "")
        set(DLCPROOT_INCLUDE_DIR "${DLCPROOT_DIR}/include/")
        set(DLCPROOT_LIB_DIR "${DLCPROOT_DIR}/lib/")
        if(EXISTS ${DLCPROOT_INCLUDE_DIR}/dl_compression.h AND EXISTS ${DLCPROOT_LIB_DIR}/libdlcomp.so)
          message(STATUS "Found DLCP: ${DLCPROOT_DIR}")
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_WEIGHT_GRAD_COMPRESSION")
          list(APPEND Caffe_LINKER_LIBS "${DLCPROOT_LIB_DIR}/libdlcomp.so")
          include_directories(SYSTEM ${DLCP_INCLUDE_DIR})
        else()
          message(STATUS "DLCP not found. DLCP_INCLUDE_DIR = ${DLCPROOT_INCLUDE_DIR} DLCP_LIB_DIR = ${DLCPROOT_LIB_DIR}")
          message(WARNING "weight grad compression is disabled.")
        endif()
      endif()
    else()
      Download_DLCP()
      if(DLCP_CXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_WEIGHT_GRAD_COMPRESSION")
        list(APPEND Caffe_LINKER_LIBS "${DLCP_LIBDIR}/libdlcomp.so")
        include_directories(SYSTEM ${DLCP_INCLDIR})
      endif()
    endif()
  endif()
endif()

# ---[ BLAS
set(MKL_EXTERNAL "0")
if(NOT APPLE)
  set(BLAS "MKL" CACHE STRING "Selected BLAS library")
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
	#--find mkl in external/mkl
	set(script_cmd "./external/mkl/prepare_mkl.sh" )
	execute_process(COMMAND ${script_cmd}
	  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	  RESULT_VARIABLE script_result
	  OUTPUT_VARIABLE RETURN_STRING)
	separate_arguments(RETURN_STRING)
	list(GET RETURN_STRING 0 MKL_ROOT_DIR)
	list(GET RETURN_STRING 1 MKL_LIBRARIES)
	list(GET RETURN_STRING 2 MKL_EXTERNAL)
	set(MKL_INCLUDE_DIR "${MKL_ROOT_DIR}/include/")
	if( ${MKL_EXTERNAL} EQUAL 1 )
	  set(MKL_LIBRARIES "${MKL_ROOT_DIR}/lib/lib${MKL_LIBRARIES}.so")
          set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH}:${MKL_ROOT_DIR}/lib/)
	else()
	  set(MKL_LIBRARIES "${MKL_ROOT_DIR}/lib/intel64/lib${MKL_LIBRARIES}.so")
          set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH}:${MKL_ROOT_DIR}/lib/intel64/)
	endif()
	message(STATUS "Found MKL: ${MKL_INCLUDE_DIR}")
	message(STATUS "Found MKL (include: ${MKL_INCLUDE_DIR}, lib: ${MKL_LIBRARIES}")	
	include_directories(SYSTEM ${MKL_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${MKL_LIBRARIES})
    add_definitions(-DUSE_MKL)
    # If MKL and OpenMP is to be used then use Intel OpenMP
    if(OPENMP_FOUND)    
      list(APPEND Caffe_LINKER_LIBS -Wl,--as-needed iomp5)
    endif()
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

# ---[ MKL2017
if(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
  if(EXISTS ${MKL_INCLUDE_DIR}/mkl_dnn.h)
    message(STATUS "Found MKL2017")
    set(MKL2017_SUPPORTED ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL2017_SUPPORTED")
    if(USE_MKL2017_AS_DEFAULT_ENGINE)
      message(STATUS "MKL2017 engine will be used as a default engine")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MKL2017_AS_DEFAULT_ENGINE")
    endif()
  else()
    message(STATUS "MKL2017 not found")
    set(MKL2017_SUPPORTED OFF)
    if(USE_MKL2017_AS_DEFAULT_ENGINE)
      message(WARNING "Flag USE_MKL2017_AS_DEFAULT_ENGINE was set, but MKL2017 not found")
    endif()
  endif()
endif()

# ---[ MKLDNN
if(DEFINED ENV{MKLDNNROOT})
  if(DISABLE_MKLDNN_DOWNLOAD)
    message(STATUS "MKLDNN download is disabled by customized setting!")
    message(STATUS "MKLDNNROOT is already set in the ENV, will use ENV MKLDNNROOT in the following compile steps!")
    add_definitions("-DDISABLE_MKLDNN_DOWNLOAD=1")
  else()
    message(STATUS "MKLDNN download is enabled by customized setting!")
    add_definitions("-DDISABLE_MKLDNN_DOWNLOAD=0")
  endif()
  set(MKLDNNROOT_DIR $ENV{MKLDNNROOT})
  if(NOT ${MKLDNNROOT_DIR} STREQUAL "")
    set(MKLDNNROOT_INCLUDE_DIR "${MKLDNNROOT_DIR}/include/")
    if(EXISTS ${MKLDNNROOT_INCLUDE_DIR}/mkldnn.hpp)
      message(STATUS "Found MKLDNN: ${MKLDNNROOT_DIR}")
      set(MKLDNN_SUPPORTED ON)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKLDNN_SUPPORTED -std=c++11")
      if(USE_MKLDNN_AS_DEFAULT_ENGINE)
        message(STATUS "MKLDNN engine will be used as a default engine")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MKLDNN_AS_DEFAULT_ENGINE")
      endif()
      list(APPEND Caffe_LINKER_LIBS "${MKLDNNROOT_DIR}/lib/libmkldnn.so")
      set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH}:${MKLDNNROOT_DIR}/lib/)
	    include_directories(SYSTEM ${MKLDNNROOT_INCLUDE_DIR})
    else()
      message(STATUS "MKLDNN not found. MKLDNN_INCLUDE_DIR = ${MKLDNNROOT_INCLUDE_DIR}")
      set(MKLDNN_SUPPORTED OFF)
      if(USE_MKLDNN_AS_DEFAULT_ENGINE)
        message(WARNING "Flag USE_MKLDNN_AS_DEFAULT_ENGINE was set, but MKLDNN not found")
      endif()
    endif()
  endif()
else()
  if(DISABLE_MKLDNN_DOWNLOAD)
    message(STATUS "MKLDNN download is disabled by customized setting!")
    message(STATUS "MKLDNNROOT is not set in the ENV, will cause compile error in the following steps!")
    add_definitions("-DDISABLE_MKLDNN_DOWNLOAD=1")
  else()
    message(STATUS "MKLDNN download is enabled by customized setting!")
    add_definitions("-DDISABLE_MKLDNN_DOWNLOAD=0")
    Download_MKLDNN()

    message(STATUS "MKLDNN will be downloaded from github and installed in ${MKLDNN_INSTALL_DIR}")
    message(STATUS "MKLDNN include directory: ${MKLDNN_INCLUDE_DIR}")
    set(MKLDNN_SUPPORTED ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKLDNN_SUPPORTED -std=c++11")
    if(USE_MKLDNN_AS_DEFAULT_ENGINE)
      message(STATUS "MKLDNN engine will be used as a default engine")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MKLDNN_AS_DEFAULT_ENGINE")
    endif()
    list(APPEND Caffe_LINKER_LIBS "${MKLDNN_INSTALL_DIR}/lib/libmkldnn.so")
    set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH}:${MKLDNN_INSTALL_DIR}/lib/)
      include_directories(${MKLDNN_INCLUDE_DIR})
  endif()
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
