if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT APPLE)
  set(bit_dest "64")
else()
  set(bit_dest "")
endif()

macro(OPTIX_find_api_library name version)
  find_library(${name}_LIBRARY
    NAMES ${name}.${version} ${name}
    PATHS "$ENV{OPTIX_DIR}/lib${bit_dest}"
    NO_DEFAULT_PATH
    )
  find_library(${name}_LIBRARY
    NAMES ${name}.${version} ${name}
    )
  if(WIN32)
    find_file(${name}_DLL
      NAMES ${name}.${version}.dll
      PATHS "$ENV{OPTIX_DIR}/bin${bit_dest}"
      NO_DEFAULT_PATH
      )
    find_file(${name}_DLL
      NAMES ${name}.${version}.dll
      )
  endif()
endmacro()

OPTIX_find_api_library(optix 1)
#OPTIX_find_api_library(optixu 1)
#OPTIX_find_api_library(optix_prime 1)

#list(APPEND OPTIX_LIBRARIES ${optix_LIBRARY} ${optixu_LIBRARY} ${optix_prime_LIBRARY})
list(APPEND OPTIX_LIBRARIES ${optix_LIBRARY})

# Include
find_path(OPTIX_INCLUDE_DIR
  NAMES optix.h
  PATHS "$ENV{OPTIX_DIR}/include"
  NO_DEFAULT_PATH
  )
find_path(OPTIX_INCLUDE_DIR
  NAMES optix.h
  )

include(FindPackageHandleStandardArgs)
#find_package_handle_standard_args(OptiX DEFAULT_MSG OPTIX_INCLUDE_DIR optix_LIBRARY optixu_LIBRARY optix_prime_LIBRARY)
find_package_handle_standard_args(OptiX DEFAULT_MSG OPTIX_INCLUDE_DIR optix_LIBRARY)

if (OPTIX_FOUND)
  caffe_parse_header(${OPTIX_INCLUDE_DIR}/optix.h
      OPTIX_VERSION_LINES OPTIX_VERSION)
  message(STATUS "Found OptiX   (include: ${OPTIX_INCLUDE_DIR}, library: ${OPTIX_LIBRARIES}, version: ${OPTIX_VERSION})")
  mark_as_advanced(OPTIX_INCLUDE_DIR OPTIX_LIBRARIES optix_LIBRARY optixu_LIBRARY optix_prime_LIBRARY)
endif()
