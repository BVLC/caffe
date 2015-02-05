SET(vecLib_INCLUDE_SEARCH_PATHS
  /System/Library/Frameworks/vecLib.framework/Versions/Current/Headers/
  /System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Versions/Current/Headers/
)

FIND_PATH(vecLib_INCLUDE_DIR NAMES cblas.h PATHS ${vecLib_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(vecLib_LIBRARY Accelerate)

SET(vecLib_FOUND ON)

MARK_AS_ADVANCED(
    vecLib_INCLUDE_DIR
    vecLib_LIBRARY
    vecLib
)

