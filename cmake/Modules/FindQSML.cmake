# This module finds the Qualcomm Snapdragon Math Libraries (QSML) libraries.

# QSML_VERSION and QSML_DIR passed as command line variables to CMAKE

set(QSML_INCLUDE_DIR)
set(QSML_LIBRARY)

message(STATUS "Checking for QSML in ${QSML_DIR}")

if(ANDROID)
  set(SEARCH_PATH NO_CMAKE_FIND_ROOT_PATH)
  set(QSML_LIB_NAME QSML-${QSML_VERSION})
else()
  set(SEARCH_PATH "")
  set(QSML_LIB_NAME QSML)
endif()


if(ANDROID)
    find_path(QSML_INCLUDE_DIR
        NAMES qsml.h
        PATHS /opt/Qualcomm/QSML-${QSML_VERSION}/android/arm64/lp64/ndk-r11/include/ ${QSML_DIR}/include
        NO_CMAKE_FIND_ROOT_PATH)

    find_library(QSML_LIBRARY
        NAMES ${QSML_LIB_NAME}
        PATHS ${CMAKE_SYSTEM_LIBRARY_PATH} /opt/Qualcomm/QSML-${QSML_VERSION}/android/arm64/lp64/ndk-r11/lib ${QSML_DIR}/lib
        NO_CMAKE_FIND_ROOT_PATH)
else()
# ARM Linux
    find_path(QSML_INCLUDE_DIR
        NAMES qsml.h
        PATHS /opt/Qualcomm/QSML-${QSML_VERSION}/linux/arm64/lp64/gcc-5.4/include ${QSML_DIR}/include
        NO_CMAKE_FIND_ROOT_PATH)

    find_library(QSML_LIBRARY
        NAMES ${QSML_LIB_NAME}
        PATHS ${CMAKE_SYSTEM_LIBRARY_PATH} /opt/Qualcomm/QSML-${QSML_VERSION}/linux/arm64/lp64/gcc-5.4/lib ${QSML_DIR}/lib
        NO_CMAKE_FIND_ROOT_PATH)
endif()

message(STATUS "Found QSML_LIBRARY: ${QSML_LIBRARY}")

if(QSML_LIBRARY AND QSML_INCLUDE_DIR)
  set(QSML_FOUND TRUE)
else()
  set(QSML_FOUND FALSE)
endif()

if(NOT QSML_FOUND AND QSML_FIND_REQUIRED)
  message(FATAL_ERROR "QSML library not found. Please specify the library location")
endif()

mark_as_advanced(QSML_INCLUDE_DIR QSML_LIBRARY)
