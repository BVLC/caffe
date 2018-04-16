#ifdef USE_LIBDNN

#include <algorithm>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/libdnn_pool_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

// Comparative check difference limit
#define kappa 0.05
// Comparative check shape size limit
#define ELEMENT_LIMIT 1000000

namespace caffe {

template <typename Dtype>
class LibDNNPoolingLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  LibDNNPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701, Caffe::GetDefaultDevice());
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~LibDNNPoolingLayerTest() {
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
    pooling_param->add_kernel_size(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int_tp num = 2;
    const int_tp channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int_tp i = 0; i < 15 * num * channels; i += 15) {
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
    LibDNNPoolingLayer<Dtype, Dtype, Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
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
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [9 5 5 8]
    //     [9 5 5 8]
    for (int_tp i = 0; i < 8 * num * channels; i += 8) {
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
      for (int_tp i = 0; i < 8 * num * channels; i += 8) {
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
    const int_tp num = 2;
    const int_tp channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2x 2 channels of:
    // [35     1     6    26    19    24]
    // [ 3    32     7    21    23    25]
    // [31     9     2    22    27    20]
    // [ 8    28    33    17    10    15]
    // [30     5    34    12    14    16]
    // [ 4    36    29    13    18    11]
    // (this is generated by magic(6) in MATLAB)
    for (int_tp i = 0; i < 36 * num * channels; i += 36) {
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
    LibDNNPoolingLayer<Dtype, Dtype, Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
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
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    32    26    27    27]
    // [32    33    33    27    27]
    // [31    34    34    27    27]
    // [36    36    34    18    18]
    for (int_tp i = 0; i < 20 * num * channels; i += 20) {
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
      for (int_tp i = 0; i < 20 * num * channels; i += 20) {
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
    const int_tp num = 2;
    const int_tp channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2x 2 channels of:
    // [35     1     6    26    19    24]
    // [ 3    32     7    21    23    25]
    // [31     9     2    22    27    20]
    // [ 8    28    33    17    10    15]
    // [30     5    34    12    14    16]
    // [ 4    36    29    13    18    11]
    // (this is generated by magic(6) in MATLAB)
    for (int_tp i = 0; i < 36 * num * channels; i += 36) {
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
    LibDNNPoolingLayer<Dtype, Dtype, Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
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
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    32    26    26]
    // [32    32    27    27]
    // [33    33    33    27]
    // [34    34    34    17]
    // [36    36    34    18]
    for (int_tp i = 0; i < 20 * num * channels; i += 20) {
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
      for (int_tp i = 0; i < 20 * num * channels; i += 20) {
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

TYPED_TEST_CASE(LibDNNPoolingLayerTest, TestDtypesFloat);

TYPED_TEST(LibDNNPoolingLayerTest, TestSetup) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

TYPED_TEST(LibDNNPoolingLayerTest, TestSetupPadded) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  pooling_param->add_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(LibDNNPoolingLayerTest, TestSetupGlobalPooling) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_global_pooling(true);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

/*
TYPED_TEST(LibDNNPoolingLayerTest, PrintBackward) {
  LayerParameter layer_param;
  layer_param.add_kernel_size(3);
  layer_param.add_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_MAX);
  LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int_tp i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_->cpu_data()[i] << endl;
  }
  for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_->cpu_data()[i] << endl;
  }

  for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = i;
  }
  layer.Backward(this->blob_top_vec_, true, this->blob_bottom_vec_);
  for (int_tp i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

TYPED_TEST(LibDNNPoolingLayerTest, TestForwardMax) {
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
}

TYPED_TEST(LibDNNPoolingLayerTest, TestForwardMaxTopMask) {
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
}

TYPED_TEST(LibDNNPoolingLayerTest, TestGradientMax) {
  for (int_tp kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int_tp kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->add_stride(2);
      pooling_param->add_pad(1);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}

TYPED_TEST(LibDNNPoolingLayerTest, TestForwardMaxPadded) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  pooling_param->add_pad(2);
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
  LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
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

TYPED_TEST(LibDNNPoolingLayerTest, TestGradientMaxTopMask) {
  for (int_tp kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int_tp kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->add_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      this->blob_top_vec_.push_back(this->blob_top_mask_);
      LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
      this->blob_top_vec_.pop_back();
    }
  }
}

TYPED_TEST(LibDNNPoolingLayerTest, TestForwardAve) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(1);
  pooling_param->add_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(TypeParam(2));
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  TypeParam epsilon = std::is_same<TypeParam, half_fp>::value ?
                  1e-3 : 1e-5;
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

TYPED_TEST(LibDNNPoolingLayerTest, TestGradientAve) {
  for (int_tp kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int_tp kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->add_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}

TYPED_TEST(LibDNNPoolingLayerTest, TestGradientAvePadded) {
  for (int_tp kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int_tp kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->add_stride(2);
      pooling_param->add_pad(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}

template<typename TypeParam>
class LibDNNPoolingLayerNDTest : public GPUDeviceTest<TypeParam> {
 protected:
  LibDNNPoolingLayerNDTest()
      : blob_bottom_(new Blob<TypeParam>()),
        blob_top_(new Blob<TypeParam>()) {
  }

  virtual void SetUp() {
    BlobShape shape;
    shape.add_dim(1);  // Batch
    shape.add_dim(8);  // Channels
    shape.add_dim(4);  // Depth
    shape.add_dim(4);  // Height
    shape.add_dim(4);  // Width
    blob_bottom_->Reshape(shape);

    shape.add_dim(1);  // Batch
    shape.add_dim(8);  // Channels
    shape.add_dim(2);  // Depth
    shape.add_dim(2);  // Height
    shape.add_dim(2);  // Width
    blob_top_->Reshape(shape);

    // fill the values
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~LibDNNPoolingLayerNDTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  void TestForward() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param =
        layer_param.mutable_pooling_param();

    pooling_param->add_kernel_size(2);
    pooling_param->add_kernel_size(2);
    pooling_param->add_kernel_size(2);

    pooling_param->add_stride(2);
    pooling_param->add_stride(2);
    pooling_param->add_stride(2);

    pooling_param->set_axis(1);

    LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    int_tp d = blob_bottom_->shape(2);
    int_tp h = blob_bottom_->shape(3);
    int_tp w = blob_bottom_->shape(4);

    TypeParam *bottom_data = blob_bottom_->mutable_cpu_data();

    vector<TypeParam> maxval(8 * 8);

    for (int_tp cd = 0; cd < d; ++cd) {
      for (int_tp ch = 0; ch < h; ++ch) {
        for (int_tp cw = 0; cw < w; ++cw) {
          for (int batch = 0; batch < 8; batch ++) {
            bottom_data[batch * 64 + cw + ch * w + cd * w * h] =
              cw + ch * w + cd * w * h;
          }
          maxval[cw/2 + (ch/2)*2 + (cd/2)*4] =
                std::max(bottom_data[cw + ch * w + cd * w * h],
                         maxval[cw/2 + (ch/2)*2 + (cd/2)*4]);
        }
      }
    }

    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const TypeParam *top_data = blob_top_->cpu_data();

    for (int i = 0; i < 2*2*2 * 8; ++i) {
      EXPECT_EQ(maxval[i % 8], top_data[i]);
    }
  }

  void TestBackward() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param =
        layer_param.mutable_pooling_param();

    pooling_param->add_kernel_size(2);
    pooling_param->add_kernel_size(2);
    pooling_param->add_kernel_size(2);

    pooling_param->add_stride(2);
    pooling_param->add_stride(2);
    pooling_param->add_stride(2);

    pooling_param->set_axis(1);

    LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    int_tp d = blob_bottom_->shape(2);
    int_tp h = blob_bottom_->shape(3);
    int_tp w = blob_bottom_->shape(4);

    TypeParam *bottom_data = blob_bottom_->mutable_cpu_data();

    vector<TypeParam> maxval(8);

    for (int_tp cd = 0; cd < d; ++cd) {
      for (int_tp ch = 0; ch < h; ++ch) {
        for (int_tp cw = 0; cw < w; ++cw) {
          bottom_data[cw + ch * w + cd * w * h] =
              cw + ch * w + cd * w * h;
            maxval[cw/2 + (ch/2)*2 + (cd/2)*4] =
                std::max(bottom_data[cw + ch * w + cd * w * h],
                         maxval[cw/2 + (ch/2)*2 + (cd/2)*4]);
        }
      }
    }

    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    TypeParam *top_diff = blob_top_->mutable_cpu_diff();
    for (int i = 0; i < 2*2*2; ++i) {
      top_diff[i] = maxval[i];
    }

    vector<bool> prop_down;
    prop_down.push_back(true);

    layer.Backward(this->blob_top_vec_, prop_down, this->blob_bottom_vec_);

    const TypeParam *bottom_diff = blob_bottom_->cpu_diff();

    for (int_tp cd = 0; cd < d; ++cd) {
      for (int_tp ch = 0; ch < h; ++ch) {
        for (int_tp cw = 0; cw < w; ++cw) {
          if (maxval[cw/2 + (ch/2)*2 + (cd/2)*4] == cw + ch * w + cd * w * h) {
            EXPECT_EQ(maxval[cw/2 + (ch/2)*2 + (cd/2)*4],
                      bottom_diff[cw + ch * w + cd * w * h]);
          } else {
            EXPECT_EQ(0, bottom_diff[cw + ch * w + cd * w * h]);
          }
        }
      }
    }
  }

  Blob<TypeParam>* const blob_bottom_;
  Blob<TypeParam>* const blob_top_;

  vector<Blob<TypeParam>*> blob_bottom_vec_;
  vector<Blob<TypeParam>*> blob_top_vec_;
};

TYPED_TEST_CASE(LibDNNPoolingLayerNDTest, TestDtypesFloat);

TYPED_TEST(LibDNNPoolingLayerNDTest, TestSetup) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param =
      layer_param.mutable_pooling_param();

  pooling_param->add_kernel_size(2);
  pooling_param->add_kernel_size(2);
  pooling_param->add_kernel_size(2);

  pooling_param->add_stride(2);
  pooling_param->add_stride(2);
  pooling_param->add_stride(2);

  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);


  LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(2, this->blob_top_->shape(2));
  EXPECT_EQ(2, this->blob_top_->shape(3));
  EXPECT_EQ(2, this->blob_top_->shape(4));
}

TYPED_TEST(LibDNNPoolingLayerNDTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(LibDNNPoolingLayerNDTest, TestBackward) {
  this->TestBackward();
}

template<typename TypeParam>
class LibDNNComparativePoolTest : public GPUDeviceTest<TypeParam> {
 protected:
  LibDNNComparativePoolTest()
      : blob_bottom_(new Blob<TypeParam>()),
        blob_bottom_ref_(new Blob<TypeParam>()),
        blob_top_(new Blob<TypeParam>()),
        blob_top_ref_(new Blob<TypeParam>()),
        rng_(rd_()) {
  }

  virtual void SetUp() {
    // fill the values
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_ref_.push_back(blob_bottom_ref_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_ref_.push_back(blob_top_ref_);
  }

  virtual ~LibDNNComparativePoolTest() {
    delete blob_bottom_;
    delete blob_bottom_ref_;
    delete blob_top_;
    delete blob_top_ref_;
  }

  bool TestForward(int_tp testIdx) {
    std::cout << "==== Test Case " << testIdx << " ====" << std::endl;

    LayerParameter layer_param;
    PoolingParameter* pooling_param =
        layer_param.mutable_pooling_param();

    std::uniform_int_distribution<int_tp> dimsRand(1, 3);

    int_tp dims = dimsRand(this->rng_);

    std::uniform_int_distribution<int_tp> dilationRand(1, 4);
    std::uniform_int_distribution<int_tp> padRand(0, 5);
    std::uniform_int_distribution<int_tp> kernelRand(2, 4);
    std::uniform_int_distribution<int_tp> strideRand(1, 5);
    std::uniform_int_distribution<int_tp> batchRand(1, 8);
    std::uniform_int_distribution<int_tp> fmapRand(1, 32);
    std::uniform_int_distribution<int_tp> poolMethodRand(0, dims != 2 ? 0 : 1);

    int_tp batchsize = batchRand(this->rng_);
    int_tp fmaps = fmapRand(this->rng_);


    // Reduce test range for compatibility with Caffe engine
    pooling_param->set_pool(
        static_cast<PoolingParameter_PoolMethod>(poolMethodRand(this->rng_)));

    std::uniform_int_distribution<int_tp> sizeRand(1,
        std::max(2, static_cast<int_tp>(pow(ELEMENT_LIMIT / (fmaps * batchsize),
                    1.0 / (static_cast<double>(dims))))));

    BlobShape shape;
    shape.add_dim(batchsize);  // Batch
    shape.add_dim(fmaps);   // Channels


    vector<int_tp> pooled_size(dims);

    for (int_tp i = 0; i < dims; ++i) {
      pooling_param->add_kernel_size(kernelRand(this->rng_));
      pooling_param->add_dilation(dilationRand(this->rng_));
      pooling_param->add_pad(std::min(static_cast<int_tp>(padRand(this->rng_)),
                      static_cast<int_tp>(pooling_param->kernel_size(i) - 1)));
      pooling_param->add_stride(strideRand(this->rng_));

      int_tp size = sizeRand(this->rng_);
      int_tp kernel_extent = pooling_param->dilation(i)
          * (pooling_param->kernel_size(i) - 1) + 1;
      size = std::max((int_tp)size,
                      (int_tp)(kernel_extent - 2 * pooling_param->pad(i)));
      shape.add_dim(size);

      pooled_size[i] = static_cast<int_tp>(ceil(
        static_cast<float>(size + 2 * pooling_param->pad(i)
            - kernel_extent) / pooling_param->stride(i))) + 1;
      if (pooling_param->pad(i) > 0) {
        // If we have padding, ensure that the last pooling starts strictly
        // inside the image (instead of at the padding);
        // otherwise clip the last.
        if ((pooled_size[i] - 1) * pooling_param->stride(i)
            >= size + pooling_param->pad(i)) {
          --pooled_size[i];
        }
        while ((pooled_size[i] - 1) * pooling_param->stride(i)
            >= size + pooling_param->pad(i) && (pooling_param->pad(i) >= 1)) {
          pooling_param->set_pad(i, pooling_param->pad(i) - 1);
        }
      }
    }


    std::cout << "Pool method: " << pooling_param->pool() << std::endl;

    std::cout << "Shape in: [";
    for (int i = 0; i < dims + 2; ++i) {
      std::cout << shape.dim(i);
      if (i < dims + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Kernel: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << pooling_param->kernel_size(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Dilation: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << pooling_param->dilation(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Stride: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << pooling_param->stride(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Pad: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << pooling_param->pad(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    blob_bottom_->Reshape(shape);
    blob_bottom_ref_->Reshape(shape);


    LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    PoolingLayer<TypeParam, TypeParam, TypeParam> ref_layer(layer_param);
    ref_layer.SetUp(this->blob_bottom_vec_ref_, this->blob_top_vec_ref_);

    for (int_tp i = 0; i < layer.blobs().size(); ++i) {
      caffe_copy(layer.blobs()[i]->count(),
                     layer.blobs()[i]->cpu_data(),
                     ref_layer.blobs()[i]->mutable_cpu_data());
    }

    caffe_rng_uniform(blob_bottom_->count(), (TypeParam)-5.0, (TypeParam)5.0,
                      blob_bottom_->mutable_cpu_data());

    caffe_copy(blob_bottom_->count(), blob_bottom_->cpu_data(),
                   blob_bottom_ref_->mutable_cpu_data());

    caffe_set(blob_top_->count(),
              (TypeParam)0.0, blob_top_->mutable_cpu_data());
    caffe_set(blob_top_ref_->count(),
              (TypeParam)0.0, blob_top_ref_->mutable_cpu_data());

    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    ref_layer.Forward(this->blob_bottom_vec_ref_, this->blob_top_vec_ref_);

    EXPECT_EQ(blob_top_->count(), blob_top_ref_->count());

    const TypeParam *top_data = blob_top_->cpu_data();
    const TypeParam *ref_top_data = blob_top_ref_->cpu_data();

    std::cout << "Shape out: [";
    for (int i = 0; i < dims + 2; ++i) {
      std::cout << blob_top_->shape()[i];
      if (i < dims + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    bool failure = false;
    double tot_error = 0;
    double tot_value = 0;
    double tot_value_ref = 0;
    int_tp failure_count = 0;

    for (int_tp i = 0; i < blob_top_->count(); ++i) {
      bool fail = (fabs(top_data[i] - ref_top_data[i]) >= kappa);
      if (fail) {
        std::cout << "Value: " << top_data[i]
                  << ", expected: " << ref_top_data[i] << " (at " << i << ")"
                  << std::endl;
        tot_error += fabs(top_data[i] - ref_top_data[i]);
        tot_value += fabs(top_data[i]);
        tot_value_ref += fabs(ref_top_data[i]);
        ++failure_count;
      }
      failure |= fail;
    }
    std::cout << "Error count: " << failure_count
              << "/" << blob_top_->count() << std::endl;
    std::cout << "Difference: " << tot_error
              << " (value: " << tot_value << " vs " << tot_value_ref << ")"
              << std::endl;

    EXPECT_EQ(failure, false);
    return failure;
  }

  bool TestBackward(int_tp testIdx) {
    std::cout << "==== Test Case " << testIdx << " ====" << std::endl;

    LayerParameter layer_param;
    PoolingParameter* pooling_param =
        layer_param.mutable_pooling_param();

    std::uniform_int_distribution<int_tp> dimsRand(1, 3);

    int_tp dims = dimsRand(this->rng_);

    std::uniform_int_distribution<int_tp> dilationRand(1, 4);
    std::uniform_int_distribution<int_tp> padRand(0, 5);
    std::uniform_int_distribution<int_tp> kernelRand(2, 4);
    std::uniform_int_distribution<int_tp> strideRand(1, 5);
    std::uniform_int_distribution<int_tp> batchRand(1, 8);
    std::uniform_int_distribution<int_tp> fmapRand(1, 32);
    std::uniform_int_distribution<int_tp> poolMethodRand(0, dims != 2 ? 0 : 1);

    int_tp batchsize = batchRand(this->rng_);
    int_tp fmaps = fmapRand(this->rng_);


    // Reduce test range for compatibility with Caffe engine
    pooling_param->set_pool(
        static_cast<PoolingParameter_PoolMethod>(poolMethodRand(this->rng_)));


    std::uniform_int_distribution<int_tp> sizeRand(1,
        std::max(2, static_cast<int_tp>(pow(ELEMENT_LIMIT / (fmaps * batchsize),
                    1.0 / (static_cast<double>(dims))))));


    BlobShape shape;
    shape.add_dim(batchsize);  // Batch
    shape.add_dim(fmaps);   // Channels


    vector<int_tp> pooled_size(dims);

    for (int_tp i = 0; i < dims; ++i) {
      pooling_param->add_kernel_size(kernelRand(this->rng_));
      pooling_param->add_dilation(dilationRand(this->rng_));
      pooling_param->add_pad(std::min(static_cast<int_tp>(padRand(this->rng_)),
                      static_cast<int_tp>(pooling_param->kernel_size(i) - 1)));
      pooling_param->add_stride(strideRand(this->rng_));

      int_tp size = sizeRand(this->rng_);
      int_tp kernel_extent = pooling_param->dilation(i)
          * (pooling_param->kernel_size(i) - 1) + 1;
      size = std::max((int_tp)size,
                      (int_tp)(kernel_extent - 2 * pooling_param->pad(i)));
      shape.add_dim(size);

      pooled_size[i] = static_cast<int_tp>(ceil(
        static_cast<float>(size + 2 * pooling_param->pad(i)
            - kernel_extent) / pooling_param->stride(i))) + 1;
      if (pooling_param->pad(i) > 0) {
        // If we have padding, ensure that the last pooling starts strictly
        // inside the image (instead of at the padding);
        // otherwise clip the last.
        if ((pooled_size[i] - 1) * pooling_param->stride(i)
            >= size + pooling_param->pad(i)) {
          --pooled_size[i];
        }
        while ((pooled_size[i] - 1) * pooling_param->stride(i)
            >= size + pooling_param->pad(i) && (pooling_param->pad(i) >= 1)) {
          pooling_param->set_pad(i, pooling_param->pad(i) - 1);
        }
      }
    }

    std::cout << "Pool method: " << pooling_param->pool() << std::endl;

    std::cout << "Shape in: [";
    for (int i = 0; i < dims + 2; ++i) {
      std::cout << shape.dim(i);
      if (i < dims + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Kernel: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << pooling_param->kernel_size(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Dilation: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << pooling_param->dilation(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Stride: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << pooling_param->stride(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Pad: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << pooling_param->pad(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    blob_bottom_->Reshape(shape);
    blob_bottom_ref_->Reshape(shape);

    LibDNNPoolingLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    PoolingLayer<TypeParam, TypeParam, TypeParam> ref_layer(layer_param);
    ref_layer.SetUp(this->blob_bottom_vec_ref_, this->blob_top_vec_ref_);

    for (int_tp i = 0; i < layer.blobs().size(); ++i) {
      caffe_copy(layer.blobs()[i]->count(),
                     layer.blobs()[i]->cpu_data(),
                     ref_layer.blobs()[i]->mutable_cpu_data());
    }

    caffe_rng_uniform(blob_top_->count(), (TypeParam)-5.0, (TypeParam)5.0,
                      blob_top_->mutable_cpu_diff());

    caffe_copy(blob_top_->count(), blob_top_->cpu_diff(),
                   blob_top_ref_->mutable_cpu_diff());

    caffe_rng_uniform(blob_bottom_->count(), (TypeParam)-5.0, (TypeParam)5.0,
                      blob_bottom_->mutable_cpu_data());

    caffe_copy(blob_bottom_->count(), blob_bottom_->cpu_data(),
                   blob_bottom_ref_->mutable_cpu_data());


    caffe_set(blob_top_->count(),  (TypeParam)0.0,
              blob_top_->mutable_cpu_data());
    caffe_set(blob_top_ref_->count(), (TypeParam)0.0,
              blob_top_ref_->mutable_cpu_data());

    caffe_set(blob_bottom_->count(),  (TypeParam)0.0,
              blob_bottom_->mutable_cpu_diff());
    caffe_set(blob_bottom_ref_->count(), (TypeParam)0.0,
              blob_bottom_ref_->mutable_cpu_diff());


    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    ref_layer.Forward(this->blob_bottom_vec_ref_, this->blob_top_vec_ref_);

    vector<bool> prop_down(1, true);

    layer.Backward(blob_top_vec_, prop_down, blob_bottom_vec_);
    ref_layer.Backward(blob_top_vec_ref_, prop_down, blob_bottom_vec_ref_);

    EXPECT_EQ(blob_bottom_->count(), blob_bottom_ref_->count());

    const TypeParam *bottom_diff = blob_bottom_->cpu_diff();
    const TypeParam *ref_bottom_diff = blob_bottom_ref_->cpu_diff();

    std::cout << "Shape out: [";
    for (int i = 0; i < dims + 2; ++i) {
      std::cout << blob_top_->shape()[i];
      if (i < dims + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    bool failure = false;
    double tot_error = 0;
    double tot_value = 0;
    double tot_value_ref = 0;
    int_tp failure_count = 0;

    for (int_tp i = 0; i < blob_bottom_->count(); ++i) {
      bool fail = (fabs(bottom_diff[i] - ref_bottom_diff[i]) >= kappa);
      if (fail) {
        std::cout << "Value: " << bottom_diff[i]
                  << ", expected: " << ref_bottom_diff[i] << " (at " << i << ")"
                  << std::endl;
        tot_error += fabs(bottom_diff[i] - ref_bottom_diff[i]);
        tot_value += fabs(bottom_diff[i]);
        tot_value_ref += fabs(ref_bottom_diff[i]);
        ++failure_count;
      }
      failure |= fail;
    }

    std::cout << "Error count: " << failure_count
        << "/" << blob_bottom_->count() << std::endl;
    std::cout << "Difference: " << tot_error
        << " (value: " << tot_value << " vs " << tot_value_ref << ")"
        << std::endl;

    EXPECT_EQ(failure, false);
    return failure;
  }

  Blob<TypeParam>* const blob_bottom_;
  Blob<TypeParam>* const blob_bottom_ref_;
  Blob<TypeParam>* const blob_top_;
  Blob<TypeParam>* const blob_top_ref_;

  vector<Blob<TypeParam>*> blob_bottom_vec_;
  vector<Blob<TypeParam>*> blob_bottom_vec_ref_;
  vector<Blob<TypeParam>*> blob_top_vec_;
  vector<Blob<TypeParam>*> blob_top_vec_ref_;

  std::random_device rd_;
  std::mt19937 rng_;
};

TYPED_TEST_CASE(LibDNNComparativePoolTest, TestDtypesFloat);

TYPED_TEST(LibDNNComparativePoolTest, TestForward) {
  for (int i = 0; i < 100; ++i) {
    if (this->TestForward(i)) {
      break;
    }
  }
}

TYPED_TEST(LibDNNComparativePoolTest, TestBackward) {
  for (int i = 0; i < 100; ++i) {
    if (this->TestBackward(i)) {
      break;
    }
  }
}

}  // namespace caffe
#endif  // USE_LIBDNN
