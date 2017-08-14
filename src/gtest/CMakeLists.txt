add_library(gtest STATIC EXCLUDE_FROM_ALL gtest.h gtest-all.cpp)
caffe_default_properties(gtest)

#add_library(gtest_main gtest_main.cc)
#target_link_libraries(gtest_main gtest)
