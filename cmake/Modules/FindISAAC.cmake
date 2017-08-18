SET(ISAAC_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/local/include
  /opt/isaac/include
  $ENV{ISAAC_HOME}
  $ENV{ISAAC_HOME}/include
  $ENV{ISAAC_HOME}/include/external
  ../isaac/include
  ../isaac/include/external
  ../../isaac/include
  ../../isaac/include/external
)

SET(ISAAC_LIB_SEARCH_PATHS
     /lib
     /lib64
     /usr/lib
     /usr/lib64
     /usr/local/lib
     /usr/local/lib64
     /opt/isaac/lib
     $ENV{ISAAC_HOME}/build/lib
     $ENV{ISAAC_HOME}/lib
     ../isaac/lib
     ../../isaac/lib
     ../isaac/build/lib
     ../../isaac/build/lib
 )

FIND_PATH(ISAAC_INCLUDE_DIR NAMES clBLAS.h PATHS ${ISAAC_INCLUDE_SEARCH_PATHS})

IF (WIN32)
  FIND_LIBRARY(ISAAC_LIBRARY
               NAMES isaac
               PATHS ${ISAAC_LIB_SEARCH_PATHS}
	       PATH_SUFFIXES Release Debug)
ELSE (WIN32)
  FIND_LIBRARY(ISAAC_LIBRARY
               NAMES isaac
               PATHS ${ISAAC_LIB_SEARCH_PATHS})
ENDIF (WIN32)

SET(ISAAC_FOUND ON)

#    Check libraries
IF(NOT ISAAC_LIBRARY)
    SET(ISAAC_FOUND OFF)
    MESSAGE(STATUS "Could not find ISAAC lib. Turning ISAAC_FOUND off")
ENDIF()

IF (ISAAC_FOUND)
  IF (NOT ISAAC_FIND_QUIETLY)
    MESSAGE(STATUS "Found ISAAC libraries: ${ISAAC_LIBRARY}")
    MESSAGE(STATUS "Found ISAAC include: ${ISAAC_INCLUDE_DIR}")
  ENDIF (NOT ISAAC_FIND_QUIETLY)
ELSE (ISAAC_FOUND)
  IF (ISAAC_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find ISAAC")
  ENDIF (ISAAC_FIND_REQUIRED)
ENDIF (ISAAC_FOUND)

MARK_AS_ADVANCED(
    ISAAC_INCLUDE_DIR
    ISAAC_LIBRARY
)

