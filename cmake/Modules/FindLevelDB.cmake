# - Find LevelDB
#
#  LevelDB_INCLUDES  - List of LevelDB includes
#  LevelDB_LIBRARIES - List of libraries when using LevelDB.
#  LevelDB_FOUND     - True if LevelDB found.

# Look for the header file.
set(LEVELDB_ROOT_INIT "$ENV{LEVELDB_ROOT}")
file(TO_CMAKE_PATH "${LEVELDB_ROOT_INIT}" LEVELDB_ROOT_INIT)
set(LEVELDB_ROOT ${LEVELDB_ROOT_INIT} CACHE PATH "LevelDB root folder")

find_path(LevelDB_INCLUDE NAMES leveldb/db.h
                          PATHS ${LEVELDB_ROOT}/include /opt/local/include /usr/local/include /usr/include
                          DOC "Path in which the file leveldb/db.h is located." )

# Look for the library.
if(MSVC)
    find_library(LevelDB_LIBRARY_RELEASE NAMES leveldb
                                 PATHS ${LEVELDB_ROOT}/lib
                                 PATH_SUFFIXES Release
                                 DOC "Path to leveldb library." )
    
    find_library(LevelDB_LIBRARY_DEBUG NAMES leveldbd leveldb
                                 PATHS ${LEVELDB_ROOT}/lib
                                 PATH_SUFFIXES Debug
                                 DOC "Path to leveldb library." )
    set(LevelDB_LIBRARY optimized ${LevelDB_LIBRARY_RELEASE} debug ${LevelDB_LIBRARY_DEBUG})
else()
    find_library(LevelDB_LIBRARY NAMES leveldb
                                 PATHS /usr/lib ${LEVELDB_ROOT}/lib
                                 DOC "Path to leveldb library." )
endif(MSVC)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LevelDB DEFAULT_MSG LevelDB_INCLUDE LevelDB_LIBRARY)

if(LEVELDB_FOUND)
  message(STATUS "Found LevelDB (include: ${LevelDB_INCLUDE}, library: ${LevelDB_LIBRARY})")
  set(LevelDB_INCLUDES ${LevelDB_INCLUDE})
  set(LevelDB_LIBRARIES ${LevelDB_LIBRARY})
  mark_as_advanced(LevelDB_INCLUDE LevelDB_LIBRARY)

  if(EXISTS "${LevelDB_INCLUDE}/leveldb/db.h")
    file(STRINGS "${LevelDB_INCLUDE}/leveldb/db.h" __version_lines
           REGEX "static const int k[^V]+Version[ \t]+=[ \t]+[0-9]+;")

    foreach(__line ${__version_lines})
      if(__line MATCHES "[^k]+kMajorVersion[ \t]+=[ \t]+([0-9]+);")
        set(LEVELDB_VERSION_MAJOR ${CMAKE_MATCH_1})
      elseif(__line MATCHES "[^k]+kMinorVersion[ \t]+=[ \t]+([0-9]+);")
        set(LEVELDB_VERSION_MINOR ${CMAKE_MATCH_1})
      endif()
    endforeach()

    if(LEVELDB_VERSION_MAJOR AND LEVELDB_VERSION_MINOR)
      set(LEVELDB_VERSION "${LEVELDB_VERSION_MAJOR}.${LEVELDB_VERSION_MINOR}")
    endif()

    caffe_clear_vars(__line __version_lines)
  endif()
endif()
