#include <string>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class HalideTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  shared_ptr<Net<Dtype> > net_;


  virtual void InitNetFromProtoString(const string& proto) {
    NetParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    net_.reset(new Net<Dtype>(param));
  }

  virtual void InitTinyNet() {
    string proto =

        "name: 'HalideTestNetwork' "
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
        "  top: 'data' "
        "  dummy_data_param { "
        "    shape { "
        "      dim: 2 "
        "      dim: 2 "
        "      dim: 3 "
        "      dim: 5 "
        "    } "
        "  } "
        "} "
        "layer { "
        "  name: 'halide' "
        "  type: 'TestFuncLayer' "
        "  bottom: 'data' "
        "  top: 'halide' "
        "  python_param { } "
        "}";

    string library = string(INSTALL_FOLDER) +
        string("/lib/halide/libtestfunc_wrappertest.so");

    proto += "external_layers : '" + library + "'";
    InitNetFromProtoString(proto);
  }

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
  }
};


typedef ::testing::Types< GPUDevice<float> > TestFloatGPU;

TYPED_TEST_CASE(HalideTest, TestFloatGPU);

TYPED_TEST(HalideTest, TestHasBlob) {
  typedef typename TypeParam::Dtype Dtype;
  this->InitTinyNet();
  const int num = 2;
  const int channels = 2;

  shared_ptr< Blob<Dtype> > blob_bottom = this->net_->blob_by_name("data");

  // Input:
  //     [1 2 5 2 3]
  //     [9 4 1 4 8]
  //     [1 2 5 2 3]
  for (int i = 0; i < 15 * num * channels; i += 15) {
    blob_bottom->mutable_cpu_data()[i +  0] = 1;
    blob_bottom->mutable_cpu_data()[i +  1] = 2;
    blob_bottom->mutable_cpu_data()[i +  2] = 5;
    blob_bottom->mutable_cpu_data()[i +  3] = 2;
    blob_bottom->mutable_cpu_data()[i +  4] = 3;
    blob_bottom->mutable_cpu_data()[i +  5] = 9;
    blob_bottom->mutable_cpu_data()[i +  6] = 4;
    blob_bottom->mutable_cpu_data()[i +  7] = 1;
    blob_bottom->mutable_cpu_data()[i +  8] = 4;
    blob_bottom->mutable_cpu_data()[i +  9] = 8;
    blob_bottom->mutable_cpu_data()[i + 10] = 1;
    blob_bottom->mutable_cpu_data()[i + 11] = 2;
    blob_bottom->mutable_cpu_data()[i + 12] = 5;
    blob_bottom->mutable_cpu_data()[i + 13] = 2;
    blob_bottom->mutable_cpu_data()[i + 14] = 3;
  }

  // Run Net;
  this->net_->Forward();

  shared_ptr< Blob<Dtype> > blob_top = this->net_->blob_by_name("halide");

  EXPECT_EQ(blob_top->num(), num);
  EXPECT_EQ(blob_top->channels(), channels);
  EXPECT_EQ(blob_top->height(), 3);
  EXPECT_EQ(blob_top->width(), 5);
  // Expected output:
  //     [  1.5   3.5   7.5   5.5   7.5]
  //     [ 10.5   6.5   4.5   8.5  13.5]
  //     [  3.5   5.5   9.5   7.5   9.5]
  for (int i = 0; i < 15 * num * channels; i += 15) {
    EXPECT_EQ(blob_top->cpu_data()[i +  0], 1.5);
    EXPECT_EQ(blob_top->cpu_data()[i +  1], 3.5);
    EXPECT_EQ(blob_top->cpu_data()[i +  2], 7.5);
    EXPECT_EQ(blob_top->cpu_data()[i +  3], 5.5);
    EXPECT_EQ(blob_top->cpu_data()[i +  4], 7.5);
    EXPECT_EQ(blob_top->cpu_data()[i +  5], 10.5);
    EXPECT_EQ(blob_top->cpu_data()[i +  6], 6.5);
    EXPECT_EQ(blob_top->cpu_data()[i +  7], 4.5);
    EXPECT_EQ(blob_top->cpu_data()[i +  8], 8.5);
    EXPECT_EQ(blob_top->cpu_data()[i +  9], 13.5);
    EXPECT_EQ(blob_top->cpu_data()[i + 10], 3.5);
    EXPECT_EQ(blob_top->cpu_data()[i + 11], 5.5);
    EXPECT_EQ(blob_top->cpu_data()[i + 12], 9.5);
    EXPECT_EQ(blob_top->cpu_data()[i + 13], 7.5);
    EXPECT_EQ(blob_top->cpu_data()[i + 14], 9.5);
  }
}


}  // namespace caffe
