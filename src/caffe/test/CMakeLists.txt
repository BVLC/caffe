#
#    
#    All test files' names must begin with a "test_" prefix
#
#
cmake_minimum_required(VERSION 2.8)
project( Test )

#    Set CUDA Device number and CMAKE_SOURCE_DIR
add_definitions(-DCMAKE_BUILD)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test_defines.hpp.in
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test_defines.hpp.cmake
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_data/sample_data_list.txt.in
    ${CMAKE_CURRENT_SOURCE_DIR}/test_data/sample_data_list.txt.cmake
)

include_directories(
        ${Caffe_SOURCE_DIR}
)

set(TEST_MAIN test_caffe_main.cpp)
list(REMOVE_ITEM TEST_SOURCES ${TEST_MAIN})


foreach(source ${TEST_SOURCES})

    get_filename_component(name ${source} NAME_WE)
    add_executable(${name}.test ${TEST_MAIN} ${source} ../blob.cpp)
    target_link_libraries(${name}.test gtest caffe)
    add_test(${name} ${name}.test)    
    
endforeach()

add_executable(test.test ${TEST_SOURCES} ${TEST_MAIN})
target_link_libraries(test.test gtest caffe)