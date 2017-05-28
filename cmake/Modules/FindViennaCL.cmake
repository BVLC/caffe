SET(ViennaCL_WITH_OPENCL TRUE)

SET(VIENNACL_INCLUDE_SEARCH_PATHS
  viennacl
  viennacl-dev
  ..
  ../viennacl
  ../viennacl-dev
  /usr/include
  /usr/local/include
  /opt/ViennaCL/include
  $ENV{VIENNACL_HOME}
  $ENV{VIENNACL_HOME}/include
)

FIND_PATH(ViennaCL_INCLUDE_DIR NAMES viennacl/forwards.h PATHS ${VIENNACL_INCLUDE_SEARCH_PATHS})

SET(ViennaCL_FOUND ON)

# Check include files
IF(NOT ViennaCL_INCLUDE_DIR)
    SET(ViennaCL_FOUND OFF)
    MESSAGE(STATUS "Could not find ViennaCL include. Turning ViennaCL_FOUND off")
ENDIF()

IF (ViennaCL_FOUND)
  IF (NOT ViennaCL_FIND_QUIETLY)
    MESSAGE(STATUS "Found ViennaCL include: ${ViennaCL_INCLUDE_DIR}")
  ENDIF (NOT ViennaCL_FIND_QUIETLY)
ELSE (ViennaCL_FOUND)
  IF (ViennaCL_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find ViennaCL")
  ENDIF (ViennaCL_FIND_REQUIRED)
ENDIF (ViennaCL_FOUND)

IF(ViennaCL_WITH_OPENCL)
  find_package(OpenCL REQUIRED)
  IF(NOT OPENCL_INCLUDE_DIRS)
    MESSAGE(FATAL_ERROR "Could not find OpenCL include.")
  ENDIF()
  MESSAGE(STATUS "Found OpenCL include: ${OPENCL_INCLUDE_DIRS}")
ENDIF(ViennaCL_WITH_OPENCL)

set(ViennaCL_INCLUDE_DIRS ${ViennaCL_INCLUDE_DIR} ${OPENCL_INCLUDE_DIRS})
set(ViennaCL_LIBRARIES ${OPENCL_LIBRARIES})

MARK_AS_ADVANCED(
  ViennaCL_INCLUDE_DIR
  ViennaCL_INCLUDE_DIRS
  ViennaCL_LIBRARIES
)
