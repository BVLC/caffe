add_library(gtest STATIC EXCLUDE_FROM_ALL gtest.h gtest-all.cpp)
caffe_default_properties(gtest)
target_include_directories(gtest PUBLIC ${Caffe_SRC_DIR})
if(NOT MSVC)
  target_compile_definitions(gtest PUBLIC -DGTEST_USE_OWN_TR1_TUPLE)
endif()


#add_library(gtest_main gtest_main.cc)
#target_link_libraries(gtest_main gtest)
