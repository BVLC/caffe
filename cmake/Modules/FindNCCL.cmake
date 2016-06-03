# Find the NCCL libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT_DIR:    Base directory where all NCCL components are found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIR
#  NCCL_LIBRARIES

find_path(NCCL_INCLUDE_DIR NAMES nccl.h PATHS ${NCCL_ROOT_DIR})

find_library(NCCL_LIBRARIES NAMES nccl PATHS ${NCCL_ROOT_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARIES)

if(NCCL_FOUND)
  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIR}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARIES)
endif()

