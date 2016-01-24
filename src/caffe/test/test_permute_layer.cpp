#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/permute_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

static const float eps = 1e-6;

template <typename TypeParam>
class PermuteLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  PermuteLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 2, 2, 3)),
      blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~PermuteLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PermuteLayerTest, TestDtypesAndDevices);

TYPED_TEST(PermuteLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PermuteParameter* permute_param = layer_param.mutable_permute_param();
  permute_param->add_order(0);
  permute_param->add_order(2);
  permute_param->add_order(3);
  permute_param->add_order(1);
  PermuteLayer<Dtype> layer(layer_param);

  this->blob_bottom_->Reshape(2, 3, 4, 5);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(PermuteLayerTest, TestSetUpIdentity) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PermuteLayer<Dtype> layer(layer_param);

  this->blob_bottom_->Reshape(2, 3, 4, 5);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(PermuteLayerTest, TestFowardIdentity) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PermuteLayer<Dtype> layer(layer_param);

  this->blob_bottom_->Reshape(2, 3, 4, 5);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_bottom_->cpu_data()[i],
                this->blob_top_->cpu_data()[i], eps);
  }
}

TYPED_TEST(PermuteLayerTest, TestFowrad2D) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PermuteParameter* permute_param = layer_param.mutable_permute_param();
  permute_param->add_order(0);
  permute_param->add_order(1);
  permute_param->add_order(3);
  permute_param->add_order(2);
  PermuteLayer<Dtype> layer(layer_param);

  const int num = 2;
  const int channels = 3;
  const int height = 2;
  const int width = 3;
  this->blob_bottom_->Reshape(num, channels, height, width);
  // Input: 2 x 3 channels of:
  //    [1 2 3]
  //    [4 5 6]
  for (int i = 0; i < height * width * num * channels; i += height * width) {
    this->blob_bottom_->mutable_cpu_data()[i + 0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i + 1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i + 2] = 3;
    this->blob_bottom_->mutable_cpu_data()[i + 3] = 4;
    this->blob_bottom_->mutable_cpu_data()[i + 4] = 5;
    this->blob_bottom_->mutable_cpu_data()[i + 5] = 6;
  }
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Expected output: 2 x 3 channels of:
  //    [1 4]
  //    [2 5]
  //    [3 6]
  for (int i = 0; i < height * width * num * channels; i += height * width) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 0], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 1], 4);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 2], 2);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 3], 5);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 4], 3);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 5], 6);
  }
}

TYPED_TEST(PermuteLayerTest, TestFowrad3D) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PermuteParameter* permute_param = layer_param.mutable_permute_param();
  permute_param->add_order(0);
  permute_param->add_order(2);
  permute_param->add_order(3);
  permute_param->add_order(1);
  PermuteLayer<Dtype> layer(layer_param);

  const int num = 2;
  const int channels = 2;
  const int height = 2;
  const int width = 3;
  this->blob_bottom_->Reshape(num, channels, height, width);
  // Input: 2 of:
  //    [1 2 3]
  //    [4 5 6]
  //    =======
  //    [7 8 9]
  //    [10 11 12]
  int inner_dim = channels * height * width;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < inner_dim; ++j) {
      this->blob_bottom_->mutable_cpu_data()[i * inner_dim + j] = j + 1;
    }
  }
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Expected output: 2 of:
  //    [1 7]
  //    [2 8]
  //    [3 9]
  //    =====
  //    [4 10]
  //    [5 11]
  //    [6 12]
  for (int i = 0; i < num * inner_dim; i += inner_dim) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 0], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 1], 7);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 2], 2);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 3], 8);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 4], 3);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 5], 9);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 6], 4);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 7], 10);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 8], 5);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 9], 11);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 10], 6);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 11], 12);
  }
}

TYPED_TEST(PermuteLayerTest, TestTwoPermute) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PermuteParameter* permute_param = layer_param.mutable_permute_param();
  permute_param->add_order(0);
  permute_param->add_order(2);
  permute_param->add_order(3);
  permute_param->add_order(1);
  PermuteLayer<Dtype> layer1(layer_param);

  Blob<Dtype> input1(2, 3, 4, 5);
  Caffe::set_random_seed(1701);
  // fill the values
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(&input1);
  Blob<Dtype> output1;
  vector<Blob<Dtype>*> bottom_vec, top_vec;
  bottom_vec.push_back(&input1);
  top_vec.push_back(&output1);
  layer1.SetUp(bottom_vec, top_vec);
  layer1.Forward(bottom_vec, top_vec);

  EXPECT_EQ(output1.num(), 2);
  EXPECT_EQ(output1.channels(), 4);
  EXPECT_EQ(output1.height(), 5);
  EXPECT_EQ(output1.width(), 3);

  // Create second permute layer which transfer back the original order.
  permute_param->clear_order();
  permute_param->add_order(0);
  permute_param->add_order(3);
  permute_param->add_order(1);
  permute_param->add_order(2);
  PermuteLayer<Dtype> layer2(layer_param);

  Blob<Dtype> output2;
  bottom_vec.clear();
  bottom_vec.push_back(&output1);
  top_vec.clear();
  top_vec.push_back(&output2);
  layer2.SetUp(bottom_vec, top_vec);
  layer2.Forward(bottom_vec, top_vec);

  EXPECT_EQ(output2.num(), 2);
  EXPECT_EQ(output2.channels(), 3);
  EXPECT_EQ(output2.height(), 4);
  EXPECT_EQ(output2.width(), 5);

  for (int i = 0; i < output2.count(); ++i) {
    EXPECT_NEAR(input1.cpu_data()[i], output2.cpu_data()[i], eps);
  }
}

TYPED_TEST(PermuteLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PermuteParameter* permute_param = layer_param.mutable_permute_param();
  permute_param->add_order(0);
  permute_param->add_order(2);
  permute_param->add_order(3);
  permute_param->add_order(1);
  PermuteLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
