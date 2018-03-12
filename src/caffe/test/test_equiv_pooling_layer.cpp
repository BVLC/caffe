#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/equiv_pooling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class EquivPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  EquivPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
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
  virtual ~EquivPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2x 2 square equivalent pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    EquivPoolingParameter* equiv_pooling_param =
        layer_param.mutable_equiv_pooling_param();
    equiv_pooling_param->set_kernel_size(2);
    equiv_pooling_param->set_stride(2);
    equiv_pooling_param->set_pool(EquivPoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 5, 5);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    //     [3 5 8 7 1]
    //     [6 2 9 1 7]
    for (int i = 0; i < 25 * num * channels; i += 25) {
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

      blob_bottom_->mutable_cpu_data()[i + 15] = 3;
      blob_bottom_->mutable_cpu_data()[i + 16] = 5;
      blob_bottom_->mutable_cpu_data()[i + 17] = 8;
      blob_bottom_->mutable_cpu_data()[i + 18] = 7;
      blob_bottom_->mutable_cpu_data()[i + 19] = 1;

      blob_bottom_->mutable_cpu_data()[i + 20] = 6;
      blob_bottom_->mutable_cpu_data()[i + 21] = 2;
      blob_bottom_->mutable_cpu_data()[i + 22] = 9;
      blob_bottom_->mutable_cpu_data()[i + 23] = 1;
      blob_bottom_->mutable_cpu_data()[i + 24] = 7;
    }
    EquivPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 3);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 3);
      EXPECT_EQ(blob_top_mask_->width(), 3);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [5 2 5]
    //     [9 7 8]
    //     [9 2 9]
    for (int i = 0; i < 9 * num * channels; i += 9) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 9);
    }
    if (blob_top_vec_.size() > 1) {
      // Expected mask output: 2x 2 channels of:
      //     [2 1 2]
      //     [5 18 9]
      //     [22 11 22]
      for (int i = 0; i < 9 * num * channels; i += 9) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  1);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  5);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4],  18);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5], 9);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6], 22);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7],  11);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 8],  22);
      }
    }
  }
};

TYPED_TEST_CASE(EquivPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(EquivPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EquivPoolingParameter* equiv_pooling_param =
      layer_param.mutable_equiv_pooling_param();
  equiv_pooling_param->set_kernel_size(3);
  equiv_pooling_param->set_stride(2);
  EquivPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(EquivPoolingLayerTest, TestSetupPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EquivPoolingParameter* equiv_pooling_param =
      layer_param.mutable_equiv_pooling_param();
  equiv_pooling_param->set_kernel_size(3);
  equiv_pooling_param->set_stride(2);
  equiv_pooling_param->set_pad(1);
  equiv_pooling_param->set_pool(EquivPoolingParameter_PoolMethod_AVE);
  EquivPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(EquivPoolingLayerTest, TestForwardMax) {
  this->TestForwardSquare();
}

TYPED_TEST(EquivPoolingLayerTest, TestForwardMaxTopMask) {
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardSquare();
}

TYPED_TEST(EquivPoolingLayerTest, TestGradientMax) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 2; kernel_h <= 3; kernel_h++) {
    for (int kernel_w = 2; kernel_w <= 3; kernel_w++) {
      // note that we only implement square kernel at present
      if (kernel_h != kernel_w) {
        continue;
      }
      LayerParameter layer_param;
      EquivPoolingParameter* equiv_pooling_param =
          layer_param.mutable_equiv_pooling_param();
      equiv_pooling_param->set_kernel_size(kernel_h);
      equiv_pooling_param->set_stride(2);
      equiv_pooling_param->set_pad(1);
      equiv_pooling_param->set_pool(EquivPoolingParameter_PoolMethod_MAX);
      EquivPoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}

TYPED_TEST(EquivPoolingLayerTest, TestForwardMaxPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EquivPoolingParameter* equiv_pooling_param =
      layer_param.mutable_equiv_pooling_param();
  equiv_pooling_param->set_kernel_size(3);
  equiv_pooling_param->set_stride(2);
  equiv_pooling_param->set_pad(2);
  equiv_pooling_param->set_pool(EquivPoolingParameter_PoolMethod_MAX);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  // Input:
  //     [ 1 2 4 ]
  //     [ 2 3 2 ]
  //     [ 4 2 1 ]
  this->blob_bottom_->mutable_cpu_data()[0] = 1;
  this->blob_bottom_->mutable_cpu_data()[1] = 2;
  this->blob_bottom_->mutable_cpu_data()[2] = 4;
  this->blob_bottom_->mutable_cpu_data()[3] = 2;
  this->blob_bottom_->mutable_cpu_data()[4] = 3;
  this->blob_bottom_->mutable_cpu_data()[5] = 2;
  this->blob_bottom_->mutable_cpu_data()[6] = 4;
  this->blob_bottom_->mutable_cpu_data()[7] = 2;
  this->blob_bottom_->mutable_cpu_data()[8] = 1;
  EquivPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-8;
  // Output:
  //     [ 4 2 4 ]
  //     [ 2 3 2 ]
  //     [ 4 2 4 ]
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 2, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 2, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 2, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 2, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 4, epsilon);
}

TYPED_TEST(EquivPoolingLayerTest, TestGradientMaxTopMask) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 2; kernel_h <= 3; kernel_h++) {
    for (int kernel_w = 2; kernel_w <= 3; kernel_w++) {
      // note that we only implement square kernel at present
      if (kernel_h != kernel_w) {
        continue;
      }
      LayerParameter layer_param;
      EquivPoolingParameter* equiv_pooling_param =
          layer_param.mutable_equiv_pooling_param();
      equiv_pooling_param->set_kernel_size(kernel_h);
      equiv_pooling_param->set_stride(2);
      equiv_pooling_param->set_pool(EquivPoolingParameter_PoolMethod_MAX);
      this->blob_top_vec_.push_back(this->blob_top_mask_);
      EquivPoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
      this->blob_top_vec_.pop_back();
    }
  }
}
}  // namespace caffe
