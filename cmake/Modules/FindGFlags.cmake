# - Try to find GFLAGS
#
# The following variables are optionally searched for defaults
#  GFLAGS_ROOT_DIR:            Base directory where all GFLAGS components are found
#
# The following are set after configuration is done:
#  GFLAGS_FOUND
#  GFLAGS_INCLUDE_DIRS
#  GFLAGS_LIBRARIES
#  GFLAGS_LIBRARYRARY_DIRS

include(FindPackageHandleStandardArgs)

set(GFLAGS_ROOT_DIR "" CACHE PATH "Folder contains Gflags")

# We are testing only a couple of files in the include directories
if(WIN32)
    find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h
        PATHS ${GFLAGS_ROOT_DIR}/include)
else()
    find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h
        PATHS ${GFLAGS_ROOT_DIR})
endif()

if(MSVC AND NOT BUILD_SHARED_LIBS)
    set(GFLAGS_LIB_NAME gflags_static)
else()
    set(GFLAGS_LIB_NAME gflags)
endif()
find_library(GFLAGS_LIBRARY ${GFLAGS_LIB_NAME})

find_package_handle_standard_args(GFlags DEFAULT_MSG GFLAGS_INCLUDE_DIR GFLAGS_LIBRARY)


if(GFLAGS_FOUND)
    set(GFLAGS_INCLUDE_DIRS ${GFLAGS_INCLUDE_DIR})
    set(GFLAGS_LIBRARIES ${GFLAGS_LIBRARY})
    message(STATUS "Found gflags  (include: ${GFLAGS_INCLUDE_DIR}, library: ${GFLAGS_LIBRARY})")
    mark_as_advanced(GFLAGS_LIBRARY GFLAGS_INCLUDE_DIR GFLAGS_ROOT_DIR)
endif()
