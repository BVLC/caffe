# Find the Accelerate libraries as part of Accelerate.framework or as standalon framework
#
# The following are set after configuration is done:
#  ACCELERATE_FOUND
#  ACCELERATE_INCLUDE_DIR
#  ACCELERATE_LINKER_LIBS


if(NOT APPLE)
  return()
endif()

set(__accelerate_include_suffix "Frameworks/Accelerate.framework/Versions/Current/Headers")

find_path(Accelerate_INCLUDE_DIR Accelerate.h
          DOC "Accelerate include directory"
          PATHS /System/Library/${__accelerate_include_suffix}
                /System/Library/Frameworks/Accelerate.framework/Versions/Current/${__accelerate_include_suffix}
                /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/Accelerate.framework/Headers/)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Accelerate DEFAULT_MSG Accelerate_INCLUDE_DIR)

if(ACCELERATE_FOUND)
  if(Accelerate_INCLUDE_DIR MATCHES "^/System/Library/Frameworks/Accelerate.framework.*")
    set(Accelerate_LINKER_LIBS -lcblas "-framework Accelerate")
    message(STATUS "Found standalone Accelerate.framework")
  else()
    set(Accelerate_LINKER_LIBS -lcblas "-framework Accelerate")
    message(STATUS "Found Accelerate as part of Accelerate.framework")
  endif()

  mark_as_advanced(Accelerate_INCLUDE_DIR)
endif()
