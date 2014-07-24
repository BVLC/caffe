#
#
#    All test files' names must begin with a "test_" prefix
#
#
project( Test )

#    Set CUDA Device number and CMAKE_SOURCE_DIR
add_definitions(-DCMAKE_BUILD)

#    Generate config files
set(IN_EXT .in)
set(GEN_EXT .gen.cmake)
set(TEST_DEFINES_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test_defines.hpp)
set(TEST_DATA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/test_data/sample_data_list.txt)

set(IN_FILES 
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
list(REMOVE_ITEM TEST_SOURCES ${TEST_MAIN})


foreach(source ${TEST_SOURCES})

    get_filename_component(name ${source} NAME_WE)
    add_executable(${name}.test ${TEST_MAIN} ${source} ../blob.cpp)
    target_link_libraries(${name}.test gtest caffe)
    #disable ctest
    #add_test(${name} ${name}.test)

    #    output dir
    set_target_properties(${name}.test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

endforeach()

add_executable(test ${TEST_SOURCES} ${TEST_MAIN})
target_link_libraries(test gtest caffe)
#    Test command
set(TEST_ARGS --gtest_shuffle)
if(CPU_ONLY)
    set(TEST_ARGS ${TEST_ARGS} --gtest_filter="-*GPU*")
endif()
add_custom_command(TARGET test POSTBUILD COMMAND test ${TEST_ARGS})

#    output dir
set_target_properties(test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

