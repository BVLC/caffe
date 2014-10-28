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
set(TEST_MAIN test_caffe_main.cpp)     # main test file (with main function)

#    Generate config files
add_definitions(-DCMAKE_BUILD)    # definition needed in order to include CMake's generated files
set(IN_EXT .in)            # generator input file extension
set(GEN_EXT .gen.cmake)    # generated output file extension
set(TEST_DEFINES_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test_defines.hpp)
set(TEST_DATA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/test_data/sample_data_list.txt)

# Function prepares name of a test executable
#    @output_name -  output variable's name
#    @filename    -  test_*.cpp file path
function(test_name output_name filename)
    get_filename_component(name ${filename} NAME_WE)
    set(${output_name} ${name}${TEST_EXT} PARENT_SCOPE)
endfunction()

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

# Remove main from test sources and prepare an Object lib with main
file(GLOB TEST_MAIN ${TEST_MAIN}) 
list(REMOVE_ITEM TEST_CPP_SOURCES ${TEST_MAIN})
add_library(main_obj EXCLUDE_FROM_ALL OBJECT ${TEST_MAIN} ../common.cpp)

#    Build each test separately from *.cpp files
foreach(source ${TEST_CPP_SOURCES})
    test_name(TEST_NAME ${source})
    
    #
    add_library(${TEST_NAME}.obj EXCLUDE_FROM_ALL OBJECT ${source})
    set(TEST_OBJ_LIB $<TARGET_OBJECTS:${TEST_NAME}.obj>)
    
    add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${TEST_OBJ_LIB} $<TARGET_OBJECTS:main_obj>)     
    target_link_libraries(${TEST_NAME} gtest caffe)
    
    #    output dir
    set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
    
    # Targets and object libs
    set(TEST_TARGETS ${TEST_TARGETS} ${TEST_NAME})
    set(TEST_OBJ_LIBS ${TEST_OBJ_LIBS} ${TEST_OBJ_LIB})
endforeach()

#    Build each test separately from *.cu files
foreach(source ${TEST_CU_SOURCES})
    test_name(TEST_NAME ${source})
    
    cuda_add_library(${TEST_NAME}.lib EXCLUDE_FROM_ALL ${source})
    
    add_executable(${TEST_NAME} EXCLUDE_FROM_ALL $<TARGET_OBJECTS:main_obj>)     
    target_link_libraries(${TEST_NAME} ${TEST_NAME}.lib gtest caffe)
    
    #    output dir
    set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
    
    # Targets and object libs
    set(TEST_TARGETS ${TEST_TARGETS} ${TEST_NAME})
    set(TEST_CU_LIBS ${TEST_CU_LIBS} ${TEST_NAME}.lib)
endforeach()

#    Build a compound test excluded from the ALL target
add_executable(${ALL_TEST} EXCLUDE_FROM_ALL ${TEST_OBJ_LIBS} $<TARGET_OBJECTS:main_obj>)
if(NOT CPU_ONLY)
    target_link_libraries(${ALL_TEST} ${TEST_CU_LIBS})
endif()
target_link_libraries(${ALL_TEST} gtest caffe)
add_dependencies(${ALL_TEST} ${TEST_TARGETS})

#    Output directory
set_target_properties(${ALL_TEST} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIRECTORY})

#    Test command
set(TEST_ARGS --gtest_shuffle)
if(CPU_ONLY)
    set(TEST_ARGS ${TEST_ARGS} --gtest_filter="-*GPU*")
endif()

add_custom_target(${RUN_TEST} COMMAND ${ALL_TEST} ${TEST_ARGS})
