

SET(clBLAS_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/local/include
  /opt/clBLAS/include
  $ENV{clBLAS_HOME}/include
)

SET(clBLAS_LIB_SEARCH_PATHS
        /lib
        /lib64
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/clBLAS/lib64
        $ENV{clBLAS_HOME}
        $ENV{clBLAS_HOME}/lib
        $ENV{clBLAS_HOME}/lib64
 )

FIND_PATH(clBLAS_INCLUDE_DIR NAMES clBLAS.h PATHS ${clBLAS_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(clBLAS_LIB NAMES clBLAS PATHS ${clBLAS_LIB_SEARCH_PATHS})

SET(clBLAS_FOUND ON)

#    Check include files
IF(NOT clBLAS_INCLUDE_DIR)
    SET(clBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find clBLAS include. Turning clBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT clBLAS_LIB)
    SET(clBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find clBLAS lib. Turning clBLAS_FOUND off")
ENDIF()

IF (clBLAS_FOUND)
  IF (NOT clBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Found clBLAS libraries: ${clBLAS_LIB}")
    MESSAGE(STATUS "Found clBLAS include: ${clBLAS_INCLUDE_DIR}")
  ENDIF (NOT clBLAS_FIND_QUIETLY)
ELSE (clBLAS_FOUND)
  IF (clBLAS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find clBLAS")
  ENDIF (clBLAS_FIND_REQUIRED)
ENDIF (clBLAS_FOUND)

MARK_AS_ADVANCED(
    clBLAS_INCLUDE_DIR
    clBLAS_LIB
    clBLAS
)

