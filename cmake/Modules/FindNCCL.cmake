# Find the NCCL libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT_DIR:    Base directory where all NCCL components are found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIR
#  NCCL_LIBRARY

find_path(NCCL_INCLUDE_DIR NAMES nccl.h
    PATHS ${NCCL_ROOT_DIR}/include
    )

find_library(NCCL_LIBRARY NAMES nccl
    PATHS ${NCCL_ROOT_DIR}/lib ${NCCL_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

if(NCCL_FOUND)
  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIR}, library: ${NCCL_LIBRARY})")
  mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
endif()

