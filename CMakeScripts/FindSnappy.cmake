# Find the Snappy libraries
#
# The following variables are optionally searched for defaults
#  Snappy_ROOT_DIR:    Base directory where all Snappy components are found
#
# The following are set after configuration is done:
#  Snappy_FOUND
#  Snappy_INCLUDE_DIRS
#  Snappy_LIBS

find_path(SNAPPY_INCLUDE_DIR
    NAMES snappy.h
    HINTS ${SNAPPY_ROOT_DIR}
          ${SNAPPY_ROOT_DIR}/include
)

find_library(SNAPPY_LIBS
    NAMES snappy
    HINTS ${SNAPPY_ROOT_DIR}
          ${SNAPPY_ROOT_DIR}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Snappy
    DEFAULT_MSG
    SNAPPY_LIBS
    SNAPPY_INCLUDE_DIR
)

mark_as_advanced(
    SNAPPY_LIBS
    SNAPPY_INCLUDE_DIR
)
