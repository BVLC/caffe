include(CMakeParseArguments)

find_program( GCOV gcov )
if(NOT GCOV)
    message(FATAL_ERROR "command gcov not found!")
endif()

set(CCOV_COMPILER_FLAGS "-g -O0 --coverage -fprofile-arcs -ftest-coverage"
    CACHE INTERNAL "")

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "Non debug build may lead to wrong code coverage result.")
endif()

function(APPEND_CCOV_COMPILER_FLAGS)
    set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${CCOV_COMPILER_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CCOV_COMPILER_FLAGS}" PARENT_SCOPE)
    message(STATUS "Setting code coverage compiler flags: ${CCOV_COMPILER_FLAGS}")
endfunction()
