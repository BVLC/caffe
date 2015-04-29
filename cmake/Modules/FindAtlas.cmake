# Find the Atlas (and Lapack) libraries
#
# The following variables are optionally searched for defaults
#  Atlas_ROOT_DIR:            Base directory where all Atlas components are found
#
# The following are set after configuration is done:
#  Atlas_FOUND
#  Atlas_INCLUDE_DIRS
#  Atlas_LIBRARIES
#  Atlas_LIBRARYRARY_DIRS

set(Atlas_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/include/atlas
  /usr/include/atlas-base
  $ENV{Atlas_ROOT_DIR}
  $ENV{Atlas_ROOT_DIR}/include
)

set(Atlas_LIB_SEARCH_PATHS
  /usr/lib/atlas
  /usr/lib/atlas-base
  /usr/lib64
  /usr/lib64/atlas
  $ENV{Atlas_ROOT_DIR}
  $ENV{Atlas_ROOT_DIR}/lib
)

find_path(Atlas_CBLAS_INCLUDE_DIR   NAMES cblas.h   PATHS ${Atlas_INCLUDE_SEARCH_PATHS})
find_path(Atlas_CLAPACK_INCLUDE_DIR NAMES clapack.h PATHS ${Atlas_INCLUDE_SEARCH_PATHS})

find_library(Atlas_CBLAS_LIBRARY  NAMES ptcblas_r ptcblas cblas_r cblas blas   PATHS ${Atlas_LIB_SEARCH_PATHS})
find_library(Atlas_BLAS_LIBRARY   NAMES atlas_r atlas tatlas satlas            PATHS ${Atlas_LIB_SEARCH_PATHS})
find_library(Atlas_LAPACK_LIBRARY NAMES alapack_r alapack lapack_atlas lapacke PATHS ${Atlas_LIB_SEARCH_PATHS})

set(LOOKED_FOR
  Atlas_CBLAS_INCLUDE_DIR
  Atlas_CLAPACK_INCLUDE_DIR

  Atlas_CBLAS_LIBRARY
  Atlas_BLAS_LIBRARY
  Atlas_LAPACK_LIBRARY
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Atlas DEFAULT_MSG ${LOOKED_FOR})

if(ATLAS_FOUND)
  set(Atlas_INCLUDE_DIR ${Atlas_CBLAS_INCLUDE_DIR} ${Atlas_CLAPACK_INCLUDE_DIR})
  set(Atlas_LIBRARIES ${Atlas_LAPACK_LIBRARY} ${Atlas_CBLAS_LIBRARY} ${Atlas_BLAS_LIBRARY})
  mark_as_advanced(${LOOKED_FOR})

  message(STATUS "Found Atlas (include: ${Atlas_CBLAS_INCLUDE_DIR}, library: ${Atlas_BLAS_LIBRARY})")
endif(ATLAS_FOUND)

