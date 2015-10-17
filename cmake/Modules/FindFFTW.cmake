# - Find fftw
# Find the native fftw includes and libraries
#
#  FFTW_INCLUDE_DIR - where to find fftw3.h, etc.
#  FFTW_LIBRARIES   - List of libraries when using fttw.
#  FFTW_FOUND       - True if fftw found.

FIND_PATH(FFTW_INCLUDE_DIR NAMES fftw3.h PATHS /opt/local/include /usr/local/include /usr/include)

FIND_LIBRARY(FFTW_LIBRARY NAMES fftw3 PATHS /usr/lib /usr/local/lib)
FIND_LIBRARY(FFTWF_LIBRARY NAMES fftw3f PATHS /usr/lib /usr/local/lib)

SET(FFTW_LIBRARIES ${FFTW_LIBRARY} ${FFTWF_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG
    FFTW_INCLUDE_DIR FFTW_LIBRARIES)

if(FFTW_FOUND)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES})
  message(STATUS "Found FFTW  (include: ${FFTW_INCLUDE_DIR}, library: ${FFTW_LIBRARIES})")
  mark_as_advanced(FFTW_INCLUDE_DIR FFTW_LIBRARIES)
endif(FFTW_FOUND)
