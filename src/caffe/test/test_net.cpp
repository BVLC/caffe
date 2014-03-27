// Copyright 2014 BVLC and contributors.

#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {


template <typename Dtype>
class NetTest : public ::testing::Test {
 protected:
  NetTest() : filename(NULL) {
  }

  virtual void SetUp() {  // Create the leveldb
    filename = tmpnam(NULL);  // get temp name
    LOG(INFO) << "Using temporary leveldb " << filename;
    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    leveldb::Status status = leveldb::DB::Open(options, filename, &db);
    CHECK(status.ok());
    for (int i = 0; i < 5; ++i) {
      Datum datum;
      datum.set_label(i);
      datum.set_channels(2);
      datum.set_height(3);
      datum.set_width(4);
      std::string* data = datum.mutable_data();
      for (int j = 0; j < 24; ++j) {
        data->push_back((uint8_t)i);
      }
      std::stringstream ss;
      ss << i;
      db->Put(leveldb::WriteOptions(), ss.str(), datum.SerializeAsString());
    }
    delete db;

    const string& proto_prefix =
        "name: 'TestNetwork' "
        "layers: { "
        "  layer { "
        "    name: 'data' "
        "    type: 'data' ";
    const string& proto_suffix =
        "    batchsize: 1 "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layers: { "
        "  layer { "
        "    name: 'innerproduct' "
        "    type: 'innerproduct' "
        "    num_output: 1000 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "    blobs_lr: 1. "
        "    blobs_lr: 2. "
        "    weight_decay: 1. "
        "    weight_decay: 0. "
        "  } "
        "  bottom: 'data' "
        "  top: 'innerproduct' "
        "} "
        "layers: { "
        "  layer { "
        "    name: 'loss' "
        "    type: 'softmax_loss' "
        "  } "
        "  bottom: 'innerproduct' "
        "  bottom: 'label' "
        "} ";
    proto = proto_prefix + "source: '" + string(this->filename) +
        "' " + proto_suffix;
  }

  virtual ~NetTest() {
  }

  char* filename;
  string proto;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(NetTest, Dtypes);

TYPED_TEST(NetTest, TestHasBlob) {
  NetParameter param;
  CHECK(google::protobuf::TextFormat::ParseFromString(this->proto,
                                                      &param));
  Net<TypeParam> net(param);
  EXPECT_TRUE(net.has_blob("data"));
  EXPECT_TRUE(net.has_blob("label"));
  EXPECT_TRUE(net.has_blob("innerproduct"));
  EXPECT_FALSE(net.has_blob("loss"));
}

TYPED_TEST(NetTest, TestGetBlob) {
  NetParameter param;
  CHECK(google::protobuf::TextFormat::ParseFromString(this->proto,
                                                      &param));
  Net<TypeParam> net(param);
  EXPECT_EQ(net.blob_by_name("data"), net.blobs()[0]);
  EXPECT_EQ(net.blob_by_name("label"), net.blobs()[1]);
  EXPECT_EQ(net.blob_by_name("innerproduct"), net.blobs()[2]);
  EXPECT_FALSE(net.blob_by_name("loss"));
}

TYPED_TEST(NetTest, TestHasLayer) {
  NetParameter param;
  CHECK(google::protobuf::TextFormat::ParseFromString(this->proto,
                                                      &param));
  Net<TypeParam> net(param);
  EXPECT_TRUE(net.has_layer("data"));
  EXPECT_TRUE(net.has_layer("innerproduct"));
  EXPECT_TRUE(net.has_layer("loss"));
  EXPECT_FALSE(net.has_layer("label"));
}

TYPED_TEST(NetTest, TestGetLayerByName) {
  NetParameter param;
  CHECK(google::protobuf::TextFormat::ParseFromString(this->proto,
                                                      &param));
  Net<TypeParam> net(param);
  EXPECT_EQ(net.layer_by_name("data"), net.layers()[0]);
  EXPECT_EQ(net.layer_by_name("innerproduct"), net.layers()[1]);
  EXPECT_EQ(net.layer_by_name("loss"), net.layers()[2]);
  EXPECT_FALSE(net.layer_by_name("label"));
}

}  // namespace caffe
