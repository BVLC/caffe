   SET(FFTW3_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/local/include
  /opt/fftw3/include
  $ENV{FFTW3_HOME}
  $ENV{FFTW3_HOME}/include
)

SET(FFTW3_LIB_SEARCH_PATHS
        /lib/
        /lib64/
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/fftw3/lib
        $ENV{FFTW3_HOME}
        $ENV{FFTW3_HOME}/lib
 )

FIND_PATH(FFTW3_INCLUDE_DIR NAMES fftw3.h PATHS ${FFTW3_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(FFTW3_LIBRARY NAMES fftw3 PATHS ${FFTW3_LIB_SEARCH_PATHS})

SET(FFTW3_FOUND ON)

#    Check include files
IF(NOT FFTW3_INCLUDE_DIR)
    SET(FFTW3_FOUND OFF)
    MESSAGE(STATUS "Could not find FFTW3 include. Turning FFTW3_FOUND off")
ENDIF()

#    Check libraries
IF(NOT FFTW3_LIBRARY)
    SET(FFTW3_FOUND OFF)
    MESSAGE(STATUS "Could not find FFTW3 lib. Turning FFTW3_FOUND off")
ENDIF()

IF (FFTW3_FOUND)
  #IF (NOT FFTW3_FIND_QUIETLY)
    MESSAGE(STATUS "Found FFTW3 libraries: ${FFTW3_LIBRARY}")
    MESSAGE(STATUS "Found FFTW3 include: ${FFTW3_INCLUDE_DIR}")
  #ENDIF (NOT FFWT3_FIND_QUIETLY)
ELSE (FFTW3_FOUND)
  IF (FFTW3_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find FFTW3")
  ENDIF (FFTW3_FIND_REQUIRED)
ENDIF (FFTW3_FOUND)

MARK_AS_ADVANCED(
    FFTW3_INCLUDE_DIR
    FFTW3_LIBRARY
)
