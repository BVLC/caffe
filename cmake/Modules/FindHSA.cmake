if (HSA_RUNTIME_INCLUDE_DIR)
  ## The HSA information is already in the cache.
  set (HSA_RUNTIME_FIND_QUIETLY TRUE)
endif (HSA_RUNTIME_INCLUDE_DIR)

## Look for the hsa include file path.

## If the HSA_INCLUDE_DIR variable is set,
## use it for the HSA_RUNTIME_INCLUDE_DIR variable.
## Otherwise set the value to /opt/hsa/include.
## Note that this can be set when running cmake
## by specifying -D HSA_INCLUDE_DIR=<directory>.

if(NOT DEFINED HSA_INCLUDE_DIR)
  set(HSA_INCLUDE_DIR
    /opt/hsa/include
    /opt/rocm/hsa/include/hsa
  )
endif()

MESSAGE("HSA_INCLUDE_DIR=${HSA_INCLUDE_DIR}")

find_path (HSA_RUNTIME_INCLUDE_DIR NAMES hsa.h PATHS ${HSA_INCLUDE_DIR})

## If the HSA_LIBRARY_DIR environment variable is set,
## use it for the HSA_RUNTIME_LIBRARY_DIR variable.
## Otherwise set the value to /opt/hsa/lib.
## Note that this can be set when running cmake
## by specifying -D HSA_LIBRARY_DIR=<directory>.

if(NOT DEFINED HSA_LIBRARY_DIR)
  set(HSA_LIBRARY_DIR
    /opt/hsa/lib
    /opt/rocm/hsa/lib
  )
endif()

MESSAGE("HSA_LIBRARY_DIR=${HSA_LIBRARY_DIR}")

## Look for the hsa library and, if found, generate the directory.
if(DEFINED CYGWIN)
    ## In CYGWIN set the library name directly to the hsa-runtime64.dll.
    ## This is a temporary work-around for cmake limitations, and requires
    ## that the HSA_RUNTIME_LIBRARY environment variable is set by the user.
    set(HSA_RUNTIME_LIBRARY "${HSA_LIBRARY_DIR}/hsa-runtime64.dll")
else()
    find_library (HSA_RUNTIME_LIBRARY NAMES hsa-runtime64 PATHS ${HSA_LIBRARY_DIR})
endif()

get_filename_component(HSA_RUNTIME_LIBRARY_DIR ${HSA_RUNTIME_LIBRARY} DIRECTORY)

## Handle the QUIETLY and REQUIRED arguments and set HSA_FOUND to TRUE if
## all listed variables are TRUE.
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (HSA "Please install 'hsa-runtime' package" HSA_RUNTIME_LIBRARY HSA_RUNTIME_INCLUDE_DIR)

if (HSA_FOUND)
    set (HSA_LIBRARIES ${HSA_LIBRARY})
else (HSA_FOUND)
    set (HSA_LIBRARIES)
endif(HSA_FOUND)

mark_as_advanced (HSA_RUNTIME_INCLUDE_DIR)
mark_as_advanced (HSA_RUNTIME_LIBRARY_DIR)
mark_as_advanced (HSA_RUNTIME_LIBRARY)