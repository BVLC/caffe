#
#
#    All test files' names must begin with a "test_" prefix
#
#
project( Test )

# Configuration
set(TEST_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)    # test executables are going to be placed there
set(TEST_EXT .testbin)           # test executable extension
set(ALL_TEST test${TEST_EXT})    # name of an executable comprising of all tests
set(RUN_TEST runtest)            # dummy target for running tests
set(TEST_MAIN_LIB test_main_lib) # name of a dummy lib containing main function

#    Generate config files
add_definitions(-DCMAKE_BUILD)    # definition needed in order to include CMake's generated files
set(IN_EXT .in)            # generator input file extension
set(GEN_EXT .gen.cmake)    # generated output file extension
set(TEST_DEFINES_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test_defines.hpp)
set(TEST_DATA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/test_data/sample_data_list.txt)

set(IN_FILES             # generator input files
    ${TEST_DEFINES_FILE}
    ${TEST_DATA_FILE}
)

foreach(in_file ${IN_FILES})    
    configure_file(
        ${in_file}${IN_EXT}
        ${in_file}${GEN_EXT}
    )    
endforeach()       

include_directories(
        ${Caffe_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set(TEST_MAIN test_caffe_main.cpp)
list(REMOVE_ITEM TEST_CPP_SOURCES ${TEST_MAIN})

if(NOT CPU_ONLY)
    set(TEST_CPP_SOURCES ${TEST_CPP_SOURCES} ${TEST_CU_SOURCES})
endif()

add_library(${TEST_MAIN_LIB} EXCLUDE_FROM_ALL ${TEST_MAIN})
target_link_libraries(${TEST_MAIN_LIB} gtest caffe)

#    Build each test separately
foreach(source ${TEST_CPP_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    set(TEST_NAME ${name}${TEST_EXT})
    
    add_library(${TEST_NAME}.lib OBJECT ${source})
    set(TEST_OBJ_LIB $<TARGET_OBJECTS:${TEST_NAME}.lib>)
    add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${TEST_OBJ_LIB}) 
    
    target_link_libraries(${TEST_NAME} ${TEST_MAIN_LIB})
    #    output dir
    set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
    
    # Targets and object libs
    set(TEST_TARGETS ${TEST_TARGETS} ${TEST_NAME})
    set(TEST_OBJ_LIBS ${TEST_OBJ_LIBS} ${TEST_OBJ_LIB})
endforeach()

#    Build a compound test excluded from the ALL target
add_executable(${ALL_TEST} EXCLUDE_FROM_ALL ${TEST_OBJ_LIBS} )
target_link_libraries(${ALL_TEST} ${TEST_MAIN_LIB})
add_dependencies(${ALL_TEST} ${TEST_TARGETS})
#    output dir
set_target_properties(${ALL_TEST} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIRECTORY})
#    Test command
set(TEST_ARGS --gtest_shuffle)
if(CPU_ONLY)
    set(TEST_ARGS ${TEST_ARGS} --gtest_filter="-*GPU*")
endif()

add_custom_target(${RUN_TEST} COMMAND ${ALL_TEST} ${TEST_ARGS})
