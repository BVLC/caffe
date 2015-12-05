#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/layers/softmax_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_softmax_layer.hpp"
#endif
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SoftmaxLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SoftmaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test sum
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        Dtype sum = 0;
        for (int j = 0; j < this->blob_top_->channels(); ++j) {
          sum += this->blob_top_->data_at(i, j, k, l);
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);
        // Test exact values
        Dtype scale = 0;
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          scale += exp(this->blob_bottom_->data_at(i, j, k, l));
        }
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          EXPECT_GE(this->blob_top_->data_at(i, j, k, l) + 1e-4,
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
              << "debug: " << i << " " << j;
          EXPECT_LE(this->blob_top_->data_at(i, j, k, l) - 1e-4,
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
              << "debug: " << i << " " << j;
        }
      }
    }
  }
}

TYPED_TEST(SoftmaxLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
}

#ifdef USE_CUDNN
template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
=======
>>>>>>> pod/device/blob.hpp
=======
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
=======
>>>>>>> pod-caffe-pod.hpp-merge
class CuDNNSoftmaxLayerTest : public ::testing::Test {
>>>>>>> origin/BVLC/parallel
=======
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
 protected:
  CuDNNSoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNSoftmaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNSoftmaxLayerTest, TestDtypes);

TYPED_TEST(CuDNNSoftmaxLayerTest, TestForwardCuDNN) {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
  Caffe::set_mode(Caffe::GPU);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
  LayerParameter layer_param;
  CuDNNSoftmaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test sum
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        TypeParam sum = 0;
        for (int j = 0; j < this->blob_top_->channels(); ++j) {
          sum += this->blob_top_->data_at(i, j, k, l);
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);
        // Test exact values
        TypeParam scale = 0;
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          scale += exp(this->blob_bottom_->data_at(i, j, k, l));
        }
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          EXPECT_GE(this->blob_top_->data_at(i, j, k, l) + 1e-4,
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
              << "debug: " << i << " " << j;
          EXPECT_LE(this->blob_top_->data_at(i, j, k, l) - 1e-4,
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
              << "debug: " << i << " " << j;
        }
      }
    }
  }
}

TYPED_TEST(CuDNNSoftmaxLayerTest, TestGradientCuDNN) {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
}

#ifdef USE_CUDNN
template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
=======
class CuDNNSoftmaxLayerTest : public ::testing::Test {
>>>>>>> origin/BVLC/parallel
=======
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
>>>>>>> caffe
=======
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
=======
<<<<<<< HEAD
<<<<<<< HEAD
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
=======
class CuDNNSoftmaxLayerTest : public ::testing::Test {
>>>>>>> origin/BVLC/parallel
=======
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
 protected:
  CuDNNSoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNSoftmaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNSoftmaxLayerTest, TestDtypes);

TYPED_TEST(CuDNNSoftmaxLayerTest, TestForwardCuDNN) {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
  Caffe::set_mode(Caffe::GPU);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  LayerParameter layer_param;
  CuDNNSoftmaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test sum
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        TypeParam sum = 0;
        for (int j = 0; j < this->blob_top_->channels(); ++j) {
          sum += this->blob_top_->data_at(i, j, k, l);
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);
        // Test exact values
        TypeParam scale = 0;
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          scale += exp(this->blob_bottom_->data_at(i, j, k, l));
        }
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          EXPECT_GE(this->blob_top_->data_at(i, j, k, l) + 1e-4,
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
              << "debug: " << i << " " << j;
          EXPECT_LE(this->blob_top_->data_at(i, j, k, l) - 1e-4,
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
              << "debug: " << i << " " << j;
        }
      }
    }
  }
}

TYPED_TEST(CuDNNSoftmaxLayerTest, TestGradientCuDNN) {
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  Caffe::set_mode(Caffe::GPU);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
  LayerParameter layer_param;
  CuDNNSoftmaxLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

#endif

}  // namespace caffe
