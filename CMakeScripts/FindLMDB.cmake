# Try to find the LMBD libraries and headers
#  LMDB_FOUND - system has LMDB lib
#  LMDB_INCLUDE_DIR - the LMDB include directory
#  LMDB_LIBRARIES - Libraries needed to use LMDB

# FindCWD based on FindGMP by:
# Copyright (c) 2006, Laurent Montel, <montel@kde.org>
#
# Redistribution and use is allowed according to the terms of the BSD license.

# Adapted from FindCWD by:
# Copyright 2013 Conrad Steenberg <conrad.steenberg@gmail.com>
# Aug 31, 2013

if (LMDB_INCLUDE_DIR AND LMDB_LIBRARIES)
  # Already in cache, be silent
  set(LMDB_FIND_QUIETLY TRUE)
endif (LMDB_INCLUDE_DIR AND LMDB_LIBRARIES)

find_path(LMDB_INCLUDE_DIR NAMES "lmdb.h" HINTS "$ENV{LMDB_DIR}/include")
find_library(LMDB_LIBRARIES NAMES lmdb HINTS $ENV{LMDB_DIR}/lib )
MESSAGE(STATUS "LMDB lib: " ${LMDB_LIBRARIES} )
MESSAGE(STATUS "LMDB include: " ${LMDB_INCLUDE} )

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LMDB DEFAULT_MSG LMDB_INCLUDE_DIR LMDB_LIBRARIES)

mark_as_advanced(LMDB_INCLUDE_DIR LMDB_LIBRARIES)
