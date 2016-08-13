#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/halide_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class HalideLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HalideLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~HalideLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  
  void TestForwardDemo() {
    LayerParameter layer_param;
    HalideParameter* halide_param = layer_param.mutable_halide_param();
    string library("/home/hasimir/lang/caffe/build/install/bin/libplip_wrapper.so");
    halide_param->set_library(library);

    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 9;
      blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      blob_bottom_->mutable_cpu_data()[i +  7] = 1;
      blob_bottom_->mutable_cpu_data()[i +  8] = 4;
      blob_bottom_->mutable_cpu_data()[i +  9] = 8;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }
    HalideLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 5);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output:
    //     [  1.5   3.5   7.5   5.5   7.5]
    //     [ 10.5   6.5   4.5   8.5  13.5]
    //     [  3.5   5.5   9.5   7.5   9.5]
   for (int i = 0; i < 8 * num * channels; i += 8) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 1.5);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 3.5);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 7.5);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 5.5);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 7.5);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 10.5);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 6.5);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 4.5);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 8.5);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 13.5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 3.5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 5.5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 9.5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 7.5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 9.5);
    }
  }
};

TYPED_TEST_CASE(HalideLayerTest, TestDtypesAndDevices);

TYPED_TEST(HalideLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HalideParameter* halide_param = layer_param.mutable_halide_param();
  string library("/home/hasimir/lang/caffe/build/install/bin/libplip_wrapper.so");
  halide_param->set_library(library);

  HalideLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 5);
}
}  // namespace caffe
