# - Find sndfile
# Find the native sndfile includes and libraries
#
#  SNDFILE_INCLUDE_DIR - where to find sndfile.h, etc.
#  SNDFILE_LIBRARIES   - List of libraries when using libsndfile.
#  SNDFILE_FOUND       - True if libsndfile found.

FIND_PATH(SNDFILE_INCLUDE_DIR NAMES sndfile.h PATHS $ENV{LEVELDB_ROOT}/include /opt/local/include /usr/local/include /usr/include)

FIND_LIBRARY(SNDFILE_LIBRARIES NAMES sndfile PATHS /usr/local/lib /usr/lib $ENV{LEVELDB_ROOT}/lib)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SNDFILE DEFAULT_MSG
    SNDFILE_INCLUDE_DIR SNDFILE_LIBRARIES)

if(SNDFILE_FOUND)
  message(STATUS "Found LibSndFile  (include: ${SNDFILE_INCLUDE_DIR}, library: ${SNDFILE_LIBRARIES})")
  mark_as_advanced(SNDFILE_INCLUDE_DIR SNDFILE_LIBRARIES)
endif()
