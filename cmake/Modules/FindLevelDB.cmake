# - Find LevelDB
#
#  LevelDB_INCLUDES  - List of LevelDB includes
#  LevelDB_LIBRARIES - List of libraries when using LevelDB.
#  LevelDB_FOUND     - True if LevelDB found.

# Look for the header file.
find_path(LevelDB_INCLUDE NAMES leveldb/db.h
                          PATHS $ENV{LEVELDB_ROOT}/include /opt/local/include /usr/local/include /usr/include
                          DOC "Path in which the file leveldb/db.h is located." )

# Look for the library.
find_library(LevelDB_LIBRARY NAMES leveldb
                             PATHS /usr/lib $ENV{LEVELDB_ROOT}/lib
                             DOC "Path to leveldb library." )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LevelDB DEFAULT_MSG LevelDB_INCLUDE LevelDB_LIBRARY)

if(LEVELDB_FOUND)
  message(STATUS "Found LevelDB (include: ${LevelDB_INCLUDE}, library: ${LevelDB_LIBRARY})")
  set(LevelDB_INCLUDES ${LevelDB_INCLUDE})
  set(LevelDB_LIBRARIES ${LevelDB_LIBRARY})
  mark_as_advanced(LevelDB_INCLUDE LevelDB_LIBRARY)
endif()
