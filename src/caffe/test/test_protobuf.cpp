// Copyright 2014 BVLC and contributors.

// This is simply a script that tries serializing protocol buffer in text
// format. Nothing special here and no actual code is being tested.
#include <string>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

class ProtoTest : public ::testing::Test {};

TEST_F(ProtoTest, TestSerialization) {
  LayerParameter param;
  param.set_name("test");
  param.set_type(LayerParameter_LayerType_NONE);
  std::cout << "Printing in binary format." << std::endl;
  std::cout << param.SerializeAsString() << std::endl;
  std::cout << "Printing in text format." << std::endl;
  std::string str;
  google::protobuf::TextFormat::PrintToString(param, &str);
  std::cout << str << std::endl;
  EXPECT_TRUE(true);
}

}  // namespace caffe
