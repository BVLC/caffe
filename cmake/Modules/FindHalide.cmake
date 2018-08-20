# Find the Halide library
#
# The following variables are optionally searched for defaults
#  HALIDE_ROOT_DIR:            Base directory where all Halide components are found
#
# The following are set after configuration is done:
#  HALIDE_FOUND
#  HALIDE_INCLUDE_DIRS
#  HALIDE_LIBRARIES
#  HALIDE_LIBRARYRARY_DIRS

find_path(HALIDE_ROOT_DIR
    NAMES include/Halide.h include/HalideRuntime.h
)

find_library(HALIDE_LIBRARIES
    NAMES Halide
    HINTS "${HALIDE_ROOT_DIR}/lib"
)

find_path(HALIDE_INCLUDE_DIR
    NAMES Halide.h HalideRuntime.h
    HINTS ${HALIDE_ROOT_DIR}/include
)

set(HALIDE_LIBRARY HALIDE_LIBRARIES)
set(HALIDE_INCLUDE_DIRS HALIDE_INCLUDE_DIR)

set(LOOKED_FOR
  HALIDE_ROOT_DIR
  HALIDE_LIBRARY
  HALIDE_LIBRARIES
  HALIDE_INCLUDE_DIR
  HALIDE_INCLUDE_DIRS
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Halide DEFAULT_MSG ${LOOKED_FOR})

if(HALIDE_FOUND)
  mark_as_advanced(${LOOKED_FOR})
  message(STATUS "Found Halide (include: ${HALIDE_INCLUDE_DIR}, library: ${HALIDE_LIBRARIES})")
endif(HALIDE_FOUND)
