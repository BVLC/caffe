// Copyright 2014 BVLC and contributors.

#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class PoolingLayerTest : public ::testing::Test {
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
  // Test for 2 x 2 square pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2 x 2 channels of:
    Dtype data[] = {1, 2, 5, 2, 3,
        9, 4, 1, 4, 8,
        1, 2, 5, 2, 3};
    for (int i = 0; i < 15 * num * channels; i += 15) {
      for (int j = 0; j < 15; ++j) {
        blob_bottom_->mutable_cpu_data()[i +  j] = data[j];
      }
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
    // Expected output: 2 x 2 channels of:
    Dtype output[] = {9, 5, 5, 8,
        9, 5, 5, 8};
    for (int i = 0; i < 8 * num * channels; i += 8) {
      for (int j = 0; j < 8; ++j) {
        EXPECT_EQ(blob_top_->cpu_data()[i + j], output[j]);
      }
    }
    if (blob_top_vec_.size() > 1) {
      // Expected mask output: 2 x 2 channels of:
      Dtype mask[] = {5, 2, 2, 9,
          5, 12, 12, 9};
      for (int i = 0; i < 8 * num * channels; i += 8) {
        for (int j = 0; j < 8; ++j) {
          EXPECT_EQ(blob_top_mask_->cpu_data()[i + j],  mask[j]);
        }
      }
    }
  }

  void TestForwardSquareFloat() {
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2 x 2 channels of:
    Dtype data[] = {1, 2, 5, 2, 3,
        9, 4, 1, 4, 8,
        1, 2, 5, 2, 3};
    for (int i = 0; i < 15 * num * channels; i += 15) {
      for (int j = 0; j < 15; ++j) {
        blob_bottom_->mutable_cpu_data()[i +  j] = data[j];
      }
    }
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(1.7);
    pooling_param->set_stride(1.7);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    // kernel: 1.7 * 1.7, stride: 1.7 * 1.7
    // x1, y1, x2, y2
    // 0, 0, 2, 2
    // 1, 0, 4, 2
    // 3, 0, 6, 2
    // 0, 1, 2, 4
    // 1, 1, 4, 4
    // 3, 1, 5, 4
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 2);
    EXPECT_EQ(blob_top_->width(), 3);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 2);
      EXPECT_EQ(blob_top_mask_->width(), 3);
    }
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2 x 2 channels of:
    Dtype output[] = {9, 5, 8,
        9, 5, 8};
    for (int i = 0; i < 6 * num * channels; i += 6) {
      for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(blob_top_->cpu_data()[i + j], output[j]);
      }
    }
    if (blob_top_vec_.size() > 1) {
      // Expected mask output: 2 x 2 channels of:
      Dtype mask[] = {5, 2, 9,
          5, 12, 9};
      for (int i = 0; i < 6 * num * channels; i += 6) {
        for (int j = 0; j < 6; ++j) {
          EXPECT_EQ(blob_top_mask_->cpu_data()[i + j],  mask[j]);
        }
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
    // Input: 2 x 2 channels of:
    // (this is generated by magic(6) in MATLAB)
    Dtype data[] = {35, 1, 6, 26, 19, 24,
        3, 32, 7, 21, 23, 25,
        31, 9, 2, 22, 27, 20,
        8, 28, 33, 17, 10, 15,
        30, 5, 34, 12, 14, 16,
        4, 36, 29, 13, 18, 11};
    for (int i = 0; i < 36 * num * channels; i += 36) {
      for (int j = 0; j < 36; ++j) {
        blob_bottom_->mutable_cpu_data()[i +  j] = data[j];
      }
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
    // Expected output: 2 x 2 channels of:
    Dtype output[] = {35, 32, 26, 27, 27,
        32, 33, 33, 27, 27,
        31, 34, 34, 27, 27,
        36, 36, 34, 18, 18};
    for (int i = 0; i < 20 * num * channels; i += 20) {
      for (int j = 0; j < 20; ++j) {
        EXPECT_EQ(blob_top_->cpu_data()[i + j], output[j]);
      }
    }
    if (blob_top_vec_.size() > 1) {
      Dtype mask[] = {0, 7, 3, 16, 16,
          7, 20, 20, 16, 16,
          12, 26, 26, 16, 16,
          31, 31, 26, 34, 34};
      for (int i = 0; i < 20 * num * channels; i += 20) {
        for (int j = 0; j < 20; ++j) {
          EXPECT_EQ(blob_top_mask_->cpu_data()[i + j], mask[j]);
        }
      }
    }
  }

  // Test for 3x 2 rectangular pooling layer with kernel_h > kernel_w
  void TestForwardRectHighFloat() {
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2 x 2 channels of:
    Dtype data[] = {35, 1, 6, 26, 19, 24,
        3, 32, 7, 21, 23, 25,
        31, 9, 2, 22, 27, 20,
        8, 28, 33, 17, 10, 15,
        30, 5, 34, 12, 14, 16,
        4, 36, 29, 13, 18, 11};
    for (int i = 0; i < 36 * num * channels; i += 36) {
      for (int j = 0; j < 36; ++j) {
        blob_bottom_->mutable_cpu_data()[i +  j] = data[j];
      }
    }
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_h(3.5);
    pooling_param->set_kernel_w(2.5);
    pooling_param->set_stride_h(3.1);
    pooling_param->set_stride_w(2.1);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    // kernel h * w: 3.5 * 2.5, stride h * w: 3.1 * 2.1
    // x1, y1, x2, y2
    // 0, 0, 3, 4
    // 2, 0, 5, 4
    // 4, 0, 7, 4
    // 0, 3, 3, 7
    // 2, 3, 5, 7
    // 4, 3, 7, 7
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 2);
    EXPECT_EQ(blob_top_->width(), 3);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 2);
      EXPECT_EQ(blob_top_mask_->width(), 3);
    }
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2 x 2 channels of:
    Dtype output[] = {35, 33, 27,
        36, 34, 18};
    for (int i = 0; i < 6 * num * channels; i += 6) {
      for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(blob_top_->cpu_data()[i + j], output[j]);
      }
    }
    if (blob_top_vec_.size() > 1) {
      Dtype mask[] = {0, 20, 16,
          31, 26, 34};
      for (int i = 0; i < 6 * num * channels; i += 6) {
        for (int j = 0; j < 6; ++j) {
          EXPECT_EQ(blob_top_mask_->cpu_data()[i + j], mask[j]);
        }
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
    // Input: 2 x 2 channels of:
    Dtype data[] = {35, 1, 6, 26, 19, 24,
        3, 32, 7, 21, 23, 25,
        31, 9, 2, 22, 27, 20,
        8, 28, 33, 17, 10, 15,
        30, 5, 34, 12, 14, 16,
        4, 36, 29, 13, 18, 11};
    for (int i = 0; i < 36 * num * channels; i += 36) {
      for (int j = 0; j < 36; ++j) {
        blob_bottom_->mutable_cpu_data()[i +  j] = data[j];
      }
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
    Dtype output[] = {35, 32, 26, 26,
        32, 32, 27, 27,
        33, 33, 33, 27,
        34, 34, 34, 17,
        36, 36, 34, 18};
    for (int i = 0; i < 20 * num * channels; i += 20) {
      for (int j = 0; j < 20; ++j) {
        EXPECT_EQ(blob_top_->cpu_data()[i + j], output[j]);
      }
    }
    if (blob_top_vec_.size() > 1) {
      Dtype mask[] = {0, 7, 3, 3,
          7, 7, 16, 16,
          20, 20, 20, 16,
          26, 26, 26, 21,
          31, 31, 26, 34};
      for (int i = 0; i < 20 * num * channels; i += 20) {
        for (int j = 0; j < 20; ++j) {
          EXPECT_EQ(blob_top_mask_->cpu_data()[i + j], mask[j]);
        }
      }
    }
  }

  // Test for rectangular pooling layer with kernel_w > kernel_h
  void TestForwardRectWideFloat() {
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2 x 2 channels of:
    Dtype data[] = {35, 1, 6, 26, 19, 24,
        3, 32, 7, 21, 23, 25,
        31, 9, 2, 22, 27, 20,
        8, 28, 33, 17, 10, 15,
        30, 5, 34, 12, 14, 16,
        4, 36, 29, 13, 18, 11};
    for (int i = 0; i < 36 * num * channels; i += 36) {
      for (int j = 0; j < 36; ++j) {
        blob_bottom_->mutable_cpu_data()[i +  j] = data[j];
      }
    }
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_h(2.5);
    pooling_param->set_kernel_w(3.5);
    pooling_param->set_stride_h(2.1);
    pooling_param->set_stride_w(3.1);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    // kernel h * w: 2.5 * 3.5, stride h * w: 2.1 * 3.1
    // x1, y1, x2, y2
    // 0, 0, 4, 3
    // 3, 0, 7, 3
    // 0, 2, 4, 5
    // 3, 2, 7, 5
    // 0, 4, 4, 7
    // 3, 4, 7, 7
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 2);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 3);
      EXPECT_EQ(blob_top_mask_->width(), 2);
    }
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2 x 2 channels of:
    Dtype output[] = {35, 27,
        34, 27,
        36, 18};
    for (int i = 0; i < 6 * num * channels; i += 6) {
      for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(blob_top_->cpu_data()[i + j], output[j]);
      }
    }
    if (blob_top_vec_.size() > 1) {
      Dtype mask[] = {0, 16,
          26, 16,
          31, 34};
      for (int i = 0; i < 6 * num * channels; i += 6) {
        for (int j = 0; j < 6; ++j) {
          EXPECT_EQ(blob_top_mask_->cpu_data()[i + j], mask[j]);
        }
      }
    }
  }
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(PoolingLayerTest, Dtypes);

TYPED_TEST(PoolingLayerTest, TestSetup) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

TYPED_TEST(PoolingLayerTest, TestSetupPadded) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

/*
TYPED_TEST(PoolingLayerTest, PrintGPUBackward) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::GPU);
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
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  layer.Backward(this->blob_top_vec_, true, &(this->blob_bottom_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

/*
TYPED_TEST(PoolingLayerTest, PrintCPUBackward) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::CPU);
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

TYPED_TEST(PoolingLayerTest, TestCPUForwardMax) {
  Caffe::set_mode(Caffe::CPU);
  this->TestForwardSquare();
  this->TestForwardSquareFloat();
  this->TestForwardRectHigh();
  this->TestForwardRectHighFloat();
  this->TestForwardRectWide();
  this->TestForwardRectWideFloat();
}

TYPED_TEST(PoolingLayerTest, TestGPUForwardMax) {
  Caffe::set_mode(Caffe::GPU);
  this->TestForwardSquare();
  this->TestForwardSquareFloat();
  this->TestForwardRectHigh();
  this->TestForwardRectHighFloat();
  this->TestForwardRectWide();
  this->TestForwardRectWideFloat();
}

TYPED_TEST(PoolingLayerTest, TestCPUForwardMaxTopMask) {
  Caffe::set_mode(Caffe::CPU);
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardSquare();
  this->TestForwardSquareFloat();
  this->TestForwardRectHigh();
  this->TestForwardRectHighFloat();
  this->TestForwardRectWide();
  this->TestForwardRectWideFloat();
}

TYPED_TEST(PoolingLayerTest, TestGPUForwardMaxTopMask) {
  Caffe::set_mode(Caffe::GPU);
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardSquare();
  this->TestForwardSquareFloat();
  this->TestForwardRectHigh();
  this->TestForwardRectHighFloat();
  this->TestForwardRectWide();
  this->TestForwardRectWideFloat();
}

TYPED_TEST(PoolingLayerTest, TestCPUGradientMax) {
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pad(1);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      Caffe::set_mode(Caffe::CPU);
      PoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestGPUGradientMax) {
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pad(1);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      Caffe::set_mode(Caffe::GPU);
      PoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestCPUForwardMaxPadded) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pad(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::CPU);
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
  PoolingLayer<TypeParam> layer(layer_param);
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

TYPED_TEST(PoolingLayerTest, TestGPUForwardMaxPadded) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pad(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::GPU);
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
  PoolingLayer<TypeParam> layer(layer_param);
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

TYPED_TEST(PoolingLayerTest, TestCPUGradientMaxTopMask) {
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      this->blob_top_vec_.push_back(this->blob_top_mask_);
      Caffe::set_mode(Caffe::CPU);
      PoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
      this->blob_top_vec_.pop_back();
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestGPUGradientMaxTopMask) {
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      this->blob_top_vec_.push_back(this->blob_top_mask_);
      Caffe::set_mode(Caffe::GPU);
      PoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
      this->blob_top_vec_.pop_back();
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestCPUForwardAve) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(1);
  pooling_param->set_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  Caffe::set_mode(Caffe::CPU);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(TypeParam(2));
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  PoolingLayer<TypeParam> layer(layer_param);
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

TYPED_TEST(PoolingLayerTest, TestGPUForwardAve) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(1);
  pooling_param->set_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  Caffe::set_mode(Caffe::GPU);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(TypeParam(2));
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  PoolingLayer<TypeParam> layer(layer_param);
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

TYPED_TEST(PoolingLayerTest, TestCPUGradientAve) {
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      Caffe::set_mode(Caffe::CPU);
      PoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestGPUGradientAve) {
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      Caffe::set_mode(Caffe::GPU);
      PoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestCPUGradientAvePadded) {
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pad(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      Caffe::set_mode(Caffe::CPU);
      PoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestGPUGradientAvePadded) {
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pad(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      Caffe::set_mode(Caffe::GPU);
      PoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

}  // namespace caffe
