project(gtest CXX C)
cmake_minimum_required(VERSION 2.6.2)

add_library(gtest gtest-all.cpp)
add_library(gtest_main gtest_main.cc)
target_link_libraries(gtest_main gtest)