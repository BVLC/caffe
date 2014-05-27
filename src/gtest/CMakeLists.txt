project(gtest CXX C)
cmake_minimum_required(VERSION 2.6.2)

include_directories(
        ${Caffe_SOURCE_DIR}
)

add_library(gtest gtest-all.cpp)
add_library(gtest_main gtest_main.cc)
target_link_libraries(gtest_main gtest)