#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PoolingLayerTest()
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
  virtual ~PoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2x 2 square pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
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
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 2);
    EXPECT_EQ(blob_top_->width(), 4);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 2);
      EXPECT_EQ(blob_top_mask_->width(), 4);
    }
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [9 5 5 8]
    //     [9 5 5 8]
    for (int i = 0; i < 8 * num * channels; i += 8) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 8);
    }
    if (blob_top_vec_.size() > 1) {
      // Expected mask output: 2x 2 channels of:
      //     [5  2  2 9]
      //     [5 12 12 9]
      for (int i = 0; i < 8 * num * channels; i += 8) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  5);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  9);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4],  5);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7],  9);
      }
    }
  }
  // Test for 3x 2 rectangular pooling layer with kernel_h > kernel_w
  void TestForwardRectHigh() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_h(3);
    pooling_param->set_kernel_w(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2x 2 channels of:
    // [35     1     6    26    19    24]
    // [ 3    32     7    21    23    25]
    // [31     9     2    22    27    20]
    // [ 8    28    33    17    10    15]
    // [30     5    34    12    14    16]
    // [ 4    36    29    13    18    11]
    // (this is generated by magic(6) in MATLAB)
    for (int i = 0; i < 36 * num * channels; i += 36) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 6;
      blob_bottom_->mutable_cpu_data()[i +  3] = 26;
      blob_bottom_->mutable_cpu_data()[i +  4] = 19;
      blob_bottom_->mutable_cpu_data()[i +  5] = 24;
      blob_bottom_->mutable_cpu_data()[i +  6] = 3;
      blob_bottom_->mutable_cpu_data()[i +  7] = 32;
      blob_bottom_->mutable_cpu_data()[i +  8] = 7;
      blob_bottom_->mutable_cpu_data()[i +  9] = 21;
      blob_bottom_->mutable_cpu_data()[i + 10] = 23;
      blob_bottom_->mutable_cpu_data()[i + 11] = 25;
      blob_bottom_->mutable_cpu_data()[i + 12] = 31;
      blob_bottom_->mutable_cpu_data()[i + 13] = 9;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 22;
      blob_bottom_->mutable_cpu_data()[i + 16] = 27;
      blob_bottom_->mutable_cpu_data()[i + 17] = 20;
      blob_bottom_->mutable_cpu_data()[i + 18] = 8;
      blob_bottom_->mutable_cpu_data()[i + 19] = 28;
      blob_bottom_->mutable_cpu_data()[i + 20] = 33;
      blob_bottom_->mutable_cpu_data()[i + 21] = 17;
      blob_bottom_->mutable_cpu_data()[i + 22] = 10;
      blob_bottom_->mutable_cpu_data()[i + 23] = 15;
      blob_bottom_->mutable_cpu_data()[i + 24] = 30;
      blob_bottom_->mutable_cpu_data()[i + 25] = 5;
      blob_bottom_->mutable_cpu_data()[i + 26] = 34;
      blob_bottom_->mutable_cpu_data()[i + 27] = 12;
      blob_bottom_->mutable_cpu_data()[i + 28] = 14;
      blob_bottom_->mutable_cpu_data()[i + 29] = 16;
      blob_bottom_->mutable_cpu_data()[i + 30] = 4;
      blob_bottom_->mutable_cpu_data()[i + 31] = 36;
      blob_bottom_->mutable_cpu_data()[i + 32] = 29;
      blob_bottom_->mutable_cpu_data()[i + 33] = 13;
      blob_bottom_->mutable_cpu_data()[i + 34] = 18;
      blob_bottom_->mutable_cpu_data()[i + 35] = 11;
    }
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 4);
    EXPECT_EQ(blob_top_->width(), 5);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 4);
      EXPECT_EQ(blob_top_mask_->width(), 5);
    }
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    32    26    27    27]
    // [32    33    33    27    27]
    // [31    34    34    27    27]
    // [36    36    34    18    18]
    for (int i = 0; i < 20 * num * channels; i += 20) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 18);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 18);
    }
    if (blob_top_vec_.size() > 1) {
        // [ 1     8     4    17    17]
        // [ 8    21    21    17    17]
        // [13    27    27    17    17]
        // [32    32    27    35    35]
      for (int i = 0; i < 20 * num * channels; i += 20) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  3);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 34);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 34);
      }
    }
  }
  // Test for rectangular pooling layer with kernel_w > kernel_h
  void TestForwardRectWide() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_h(2);
    pooling_param->set_kernel_w(3);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2x 2 channels of:
    // [35     1     6    26    19    24]
    // [ 3    32     7    21    23    25]
    // [31     9     2    22    27    20]
    // [ 8    28    33    17    10    15]
    // [30     5    34    12    14    16]
    // [ 4    36    29    13    18    11]
    // (this is generated by magic(6) in MATLAB)
    for (int i = 0; i < 36 * num * channels; i += 36) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 6;
      blob_bottom_->mutable_cpu_data()[i +  3] = 26;
      blob_bottom_->mutable_cpu_data()[i +  4] = 19;
      blob_bottom_->mutable_cpu_data()[i +  5] = 24;
      blob_bottom_->mutable_cpu_data()[i +  6] = 3;
      blob_bottom_->mutable_cpu_data()[i +  7] = 32;
      blob_bottom_->mutable_cpu_data()[i +  8] = 7;
      blob_bottom_->mutable_cpu_data()[i +  9] = 21;
      blob_bottom_->mutable_cpu_data()[i + 10] = 23;
      blob_bottom_->mutable_cpu_data()[i + 11] = 25;
      blob_bottom_->mutable_cpu_data()[i + 12] = 31;
      blob_bottom_->mutable_cpu_data()[i + 13] = 9;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 22;
      blob_bottom_->mutable_cpu_data()[i + 16] = 27;
      blob_bottom_->mutable_cpu_data()[i + 17] = 20;
      blob_bottom_->mutable_cpu_data()[i + 18] = 8;
      blob_bottom_->mutable_cpu_data()[i + 19] = 28;
      blob_bottom_->mutable_cpu_data()[i + 20] = 33;
      blob_bottom_->mutable_cpu_data()[i + 21] = 17;
      blob_bottom_->mutable_cpu_data()[i + 22] = 10;
      blob_bottom_->mutable_cpu_data()[i + 23] = 15;
      blob_bottom_->mutable_cpu_data()[i + 24] = 30;
      blob_bottom_->mutable_cpu_data()[i + 25] = 5;
      blob_bottom_->mutable_cpu_data()[i + 26] = 34;
      blob_bottom_->mutable_cpu_data()[i + 27] = 12;
      blob_bottom_->mutable_cpu_data()[i + 28] = 14;
      blob_bottom_->mutable_cpu_data()[i + 29] = 16;
      blob_bottom_->mutable_cpu_data()[i + 30] = 4;
      blob_bottom_->mutable_cpu_data()[i + 31] = 36;
      blob_bottom_->mutable_cpu_data()[i + 32] = 29;
      blob_bottom_->mutable_cpu_data()[i + 33] = 13;
      blob_bottom_->mutable_cpu_data()[i + 34] = 18;
      blob_bottom_->mutable_cpu_data()[i + 35] = 11;
    }
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 5);
    EXPECT_EQ(blob_top_->width(), 4);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 5);
      EXPECT_EQ(blob_top_mask_->width(), 4);
    }
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    32    26    26]
    // [32    32    27    27]
    // [33    33    33    27]
    // [34    34    34    17]
    // [36    36    34    18]
    for (int i = 0; i < 20 * num * channels; i += 20) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 17);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 18);
    }
    if (blob_top_vec_.size() > 1) {
        // [ 1     8     4     4]
        // [ 8     8    17    17]
        // [21    21    21    17]
        // [27    27    27    22]
        // [32    32    27    35]
      for (int i = 0; i < 20 * num * channels; i += 20) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  3);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3],  3);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 21);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 34);
      }
    }
  }
};

TYPED_TEST_CASE(PoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(PoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

TYPED_TEST(PoolingLayerTest, TestSetupPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

/*
TYPED_TEST(PoolingLayerTest, PrintBackward) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_MAX);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_->cpu_data()[i] << endl;
  }

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = i;
  }
  layer.Backward(this->blob_top_vec_, true, &(this->blob_bottom_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

TYPED_TEST(PoolingLayerTest, TestForwardMax) {
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
}

TYPED_TEST(PoolingLayerTest, TestForwardMaxTopMask) {
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
}

TYPED_TEST(PoolingLayerTest, TestGradientMax) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pad(1);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      PoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestForwardMaxPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pad(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
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
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Dtype epsilon = 1e-8;
  // Output:
  //     [ 1 4 4 ]
  //     [ 4 4 4 ]
  //     [ 4 4 1 ]
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 1, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 1, epsilon);
}

TYPED_TEST(PoolingLayerTest, TestGradientMaxTopMask) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      this->blob_top_vec_.push_back(this->blob_top_mask_);
      PoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
      this->blob_top_vec_.pop_back();
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestForwardAve) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(1);
  pooling_param->set_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Dtype epsilon = 1e-5;
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 2.0    , epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 8.0 / 9, epsilon);
}

TYPED_TEST(PoolingLayerTest, TestGradientAve) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      PoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestGradientAvePadded) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pad(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      PoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNPoolingLayerTest : public ::testing::Test {
 protected:
  CuDNNPoolingLayerTest()
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
  virtual ~CuDNNPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2x 2 square pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
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
    CuDNNPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 2);
    EXPECT_EQ(blob_top_->width(), 4);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 2);
      EXPECT_EQ(blob_top_mask_->width(), 4);
    }
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [9 5 5 8]
    //     [9 5 5 8]
    for (int i = 0; i < 8 * num * channels; i += 8) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 8);
    }
    if (blob_top_vec_.size() > 1) {
      // Expected mask output: 2x 2 channels of:
      //     [5  2  2 9]
      //     [5 12 12 9]
      for (int i = 0; i < 8 * num * channels; i += 8) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  5);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  9);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4],  5);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7],  9);
      }
    }
  }
  // Test for 3x 2 rectangular pooling layer with kernel_h > kernel_w
  void TestForwardRectHigh() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_h(3);
    pooling_param->set_kernel_w(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2x 2 channels of:
    // [35     1     6    26    19    24]
    // [ 3    32     7    21    23    25]
    // [31     9     2    22    27    20]
    // [ 8    28    33    17    10    15]
    // [30     5    34    12    14    16]
    // [ 4    36    29    13    18    11]
    // (this is generated by magic(6) in MATLAB)
    for (int i = 0; i < 36 * num * channels; i += 36) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 6;
      blob_bottom_->mutable_cpu_data()[i +  3] = 26;
      blob_bottom_->mutable_cpu_data()[i +  4] = 19;
      blob_bottom_->mutable_cpu_data()[i +  5] = 24;
      blob_bottom_->mutable_cpu_data()[i +  6] = 3;
      blob_bottom_->mutable_cpu_data()[i +  7] = 32;
      blob_bottom_->mutable_cpu_data()[i +  8] = 7;
      blob_bottom_->mutable_cpu_data()[i +  9] = 21;
      blob_bottom_->mutable_cpu_data()[i + 10] = 23;
      blob_bottom_->mutable_cpu_data()[i + 11] = 25;
      blob_bottom_->mutable_cpu_data()[i + 12] = 31;
      blob_bottom_->mutable_cpu_data()[i + 13] = 9;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 22;
      blob_bottom_->mutable_cpu_data()[i + 16] = 27;
      blob_bottom_->mutable_cpu_data()[i + 17] = 20;
      blob_bottom_->mutable_cpu_data()[i + 18] = 8;
      blob_bottom_->mutable_cpu_data()[i + 19] = 28;
      blob_bottom_->mutable_cpu_data()[i + 20] = 33;
      blob_bottom_->mutable_cpu_data()[i + 21] = 17;
      blob_bottom_->mutable_cpu_data()[i + 22] = 10;
      blob_bottom_->mutable_cpu_data()[i + 23] = 15;
      blob_bottom_->mutable_cpu_data()[i + 24] = 30;
      blob_bottom_->mutable_cpu_data()[i + 25] = 5;
      blob_bottom_->mutable_cpu_data()[i + 26] = 34;
      blob_bottom_->mutable_cpu_data()[i + 27] = 12;
      blob_bottom_->mutable_cpu_data()[i + 28] = 14;
      blob_bottom_->mutable_cpu_data()[i + 29] = 16;
      blob_bottom_->mutable_cpu_data()[i + 30] = 4;
      blob_bottom_->mutable_cpu_data()[i + 31] = 36;
      blob_bottom_->mutable_cpu_data()[i + 32] = 29;
      blob_bottom_->mutable_cpu_data()[i + 33] = 13;
      blob_bottom_->mutable_cpu_data()[i + 34] = 18;
      blob_bottom_->mutable_cpu_data()[i + 35] = 11;
    }
    CuDNNPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 4);
    EXPECT_EQ(blob_top_->width(), 5);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 4);
      EXPECT_EQ(blob_top_mask_->width(), 5);
    }
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    32    26    27    27]
    // [32    33    33    27    27]
    // [31    34    34    27    27]
    // [36    36    34    18    18]
    for (int i = 0; i < 20 * num * channels; i += 20) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 18);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 18);
    }
    if (blob_top_vec_.size() > 1) {
        // [ 1     8     4    17    17]
        // [ 8    21    21    17    17]
        // [13    27    27    17    17]
        // [32    32    27    35    35]
      for (int i = 0; i < 20 * num * channels; i += 20) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  3);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 34);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 34);
      }
    }
  }
  // Test for rectangular pooling layer with kernel_w > kernel_h
  void TestForwardRectWide() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_h(2);
    pooling_param->set_kernel_w(3);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2x 2 channels of:
    // [35     1     6    26    19    24]
    // [ 3    32     7    21    23    25]
    // [31     9     2    22    27    20]
    // [ 8    28    33    17    10    15]
    // [30     5    34    12    14    16]
    // [ 4    36    29    13    18    11]
    // (this is generated by magic(6) in MATLAB)
    for (int i = 0; i < 36 * num * channels; i += 36) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 6;
      blob_bottom_->mutable_cpu_data()[i +  3] = 26;
      blob_bottom_->mutable_cpu_data()[i +  4] = 19;
      blob_bottom_->mutable_cpu_data()[i +  5] = 24;
      blob_bottom_->mutable_cpu_data()[i +  6] = 3;
      blob_bottom_->mutable_cpu_data()[i +  7] = 32;
      blob_bottom_->mutable_cpu_data()[i +  8] = 7;
      blob_bottom_->mutable_cpu_data()[i +  9] = 21;
      blob_bottom_->mutable_cpu_data()[i + 10] = 23;
      blob_bottom_->mutable_cpu_data()[i + 11] = 25;
      blob_bottom_->mutable_cpu_data()[i + 12] = 31;
      blob_bottom_->mutable_cpu_data()[i + 13] = 9;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 22;
      blob_bottom_->mutable_cpu_data()[i + 16] = 27;
      blob_bottom_->mutable_cpu_data()[i + 17] = 20;
      blob_bottom_->mutable_cpu_data()[i + 18] = 8;
      blob_bottom_->mutable_cpu_data()[i + 19] = 28;
      blob_bottom_->mutable_cpu_data()[i + 20] = 33;
      blob_bottom_->mutable_cpu_data()[i + 21] = 17;
      blob_bottom_->mutable_cpu_data()[i + 22] = 10;
      blob_bottom_->mutable_cpu_data()[i + 23] = 15;
      blob_bottom_->mutable_cpu_data()[i + 24] = 30;
      blob_bottom_->mutable_cpu_data()[i + 25] = 5;
      blob_bottom_->mutable_cpu_data()[i + 26] = 34;
      blob_bottom_->mutable_cpu_data()[i + 27] = 12;
      blob_bottom_->mutable_cpu_data()[i + 28] = 14;
      blob_bottom_->mutable_cpu_data()[i + 29] = 16;
      blob_bottom_->mutable_cpu_data()[i + 30] = 4;
      blob_bottom_->mutable_cpu_data()[i + 31] = 36;
      blob_bottom_->mutable_cpu_data()[i + 32] = 29;
      blob_bottom_->mutable_cpu_data()[i + 33] = 13;
      blob_bottom_->mutable_cpu_data()[i + 34] = 18;
      blob_bottom_->mutable_cpu_data()[i + 35] = 11;
    }
    CuDNNPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 5);
    EXPECT_EQ(blob_top_->width(), 4);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 5);
      EXPECT_EQ(blob_top_mask_->width(), 4);
    }
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    32    26    26]
    // [32    32    27    27]
    // [33    33    33    27]
    // [34    34    34    17]
    // [36    36    34    18]
    for (int i = 0; i < 20 * num * channels; i += 20) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 17);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 18);
    }
    if (blob_top_vec_.size() > 1) {
        // [ 1     8     4     4]
        // [ 8     8    17    17]
        // [21    21    21    17]
        // [27    27    27    22]
        // [32    32    27    35]
      for (int i = 0; i < 20 * num * channels; i += 20) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  3);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3],  3);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 21);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 34);
      }
    }
  }
};

TYPED_TEST_CASE(CuDNNPoolingLayerTest, TestDtypes);

TYPED_TEST(CuDNNPoolingLayerTest, TestSetupCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

TYPED_TEST(CuDNNPoolingLayerTest, TestSetupPaddedCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

/*
TYPED_TEST(CuDNNPoolingLayerTest, PrintBackwardCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_MAX);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_->cpu_data()[i] << endl;
  }

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = i;
  }
  layer.Backward(this->blob_top_vec_, true, &(this->blob_bottom_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

TYPED_TEST(CuDNNPoolingLayerTest, TestForwardMaxCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
}

TYPED_TEST(CuDNNPoolingLayerTest, TestForwardMaxTopMaskCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
}

TYPED_TEST(CuDNNPoolingLayerTest, TestGradientMaxCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pad(1);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      CuDNNPoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

TYPED_TEST(CuDNNPoolingLayerTest, TestForwardMaxPaddedCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pad(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
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
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  TypeParam epsilon = 1e-8;
  // Output:
  //     [ 1 4 4 ]
  //     [ 4 4 4 ]
  //     [ 4 4 1 ]
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 1, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 1, epsilon);
}

TYPED_TEST(CuDNNPoolingLayerTest, TestGradientMaxTopMaskCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      this->blob_top_vec_.push_back(this->blob_top_mask_);
      CuDNNPoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
      this->blob_top_vec_.pop_back();
    }
  }
}

TYPED_TEST(CuDNNPoolingLayerTest, TestForwardAveCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(1);
  pooling_param->set_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(TypeParam(2));
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  TypeParam epsilon = 1e-5;
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 2.0    , epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 8.0 / 9, epsilon);
}

TYPED_TEST(CuDNNPoolingLayerTest, TestGradientAveCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      CuDNNPoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

TYPED_TEST(CuDNNPoolingLayerTest, TestGradientAvePaddedCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pad(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      CuDNNPoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

#endif

}  // namespace caffe
