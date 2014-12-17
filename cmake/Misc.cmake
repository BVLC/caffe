# ---[ Configurations types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Possible configurations" FORCE)
mark_as_advanced(CMAKE_CONFIGURATION_TYPES)

if(DEFINED CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

# --[ If user doesn't specify build type then assume release
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---[ Solution folders
caffe_option(USE_PROJECT_FOLDERS "IDE Solution folders" (MSVC_IDE OR CMAKE_GENERATOR MATCHES Xcode) )

if(USE_PROJECT_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

# ---[ Install options
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Default install path" FORCE)
endif()

# ---[ RPATH settings
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOLEAN "Use link paths for shared library rpath")

# ---[ Funny target
if(UNIX OR APPLE)
  add_custom_target(simlink_to_build COMMAND "ln" "-sf" "${CMAKE_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/build"
                                     COMMENT "Adding simlink: <caffe_root>/build -> ${CMAKE_BINARY_DIR}" )
endif()

# ---[ Set debug postfix
set(Caffe_DEBUG_POSTFIX "-d")

set(CAffe_POSTFIX "")
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CAffe_POSTFIX ${Caffe_DEBUG_POSTFIX})
endif()
