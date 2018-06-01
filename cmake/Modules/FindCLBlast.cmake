set(CLBLAST_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/local/include
  /usr/local/include/clblast
  /usr/local/include  
  clblast
  CLBlast
  ..
  ../clblast
  ../CLBlast
)

set(CLBLAST_LIBRARY_SEARCH_PATHS
  /lib/
  /lib64/
  /usr/lib
  /usr/lib/clblast
  /usr/lib64
  /usr/local/lib
  /usr/local/lib64
  clblast/build
  CLBlast/build
  ..
  ../clblast/build
  ../CLBlast/build
)

find_path(CLBLAST_INCLUDE_DIR NAMES clblast.h PATHS ${CLBLAST_INCLUDE_SEARCH_PATHS})
find_library(CLBLAST_LIBRARY NAMES clblast PATHS ${CLBLAST_LIBRARY_SEARCH_PATHS})

set(CLBLAST_FOUND ON)

#    Check include files
if(NOT CLBLAST_INCLUDE_DIR)
  set(CLBLAST_FOUND OFF)
  message(STATUS "Could not find CLBLAST include. Turning CLBLAST_FOUND off")
endif()

#    Check libraries
if(NOT CLBLAST_LIBRARY)
  set(CLBLAST_FOUND OFF)
  message(STATUS "Could not find CLBLAST lib. Turning CLBLAST_FOUND off")
endif()

if (CLBLAST_FOUND)
  if (NOT CLBLAST_FIND_QUIETLY)
    message(STATUS "Found CLBLAST libraries: ${CLBLAST_LIBRARY}")
    message(STATUS "Found CLBLAST include: ${CLBLAST_INCLUDE_DIR}")
  endif (NOT CLBLAST_FIND_QUIETLY)
else (CLBLAST_FOUND)
  if (CLBLAST_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find CLBLAST")
  endif (CLBLAST_FIND_REQUIRED)
endif (CLBLAST_FOUND)

mark_as_advanced(
  CLBLAST_INCLUDE_DIR
  CLBLAST_LIBRARY
  CLBLAST
)
