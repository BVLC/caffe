#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SoftmaxLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 4, 2 , 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    //GaussianFiller<Dtype> filler(filler_param);
    //filler.Fill(this->blob_bottom_);
    Dtype* data = blob_bottom_->mutable_cpu_data();
    const int count = blob_bottom_->count();
    for (int i = 0; i < count; ++i) {
      data[i] = 10. + (float)i;
    }
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
  SoftmaxParameter* softmax_param =
      layer_param.mutable_softmax_param();
  int slice = 2;
  softmax_param->set_slice(slice);
  SoftmaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  // Test sum
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        Dtype sum = 0;
        
        int num_elements = this->blob_bottom_->channels()/slice;
        
        for (int j = 0; j < num_elements; ++j) {
          sum += this->blob_top_->data_at(i, j, k, l);
        }
        EXPECT_GE(sum, 0.999 )<< "GE";
        EXPECT_LE(sum, 1.001) << "LE";
        // Test exact values
        Dtype scale = 0;
        
        for (int j = 0; j < num_elements; ++j) {
          scale += exp(this->blob_bottom_->data_at(i, j, k, l));
        }
        for (int j = 0; j < num_elements; ++j) {
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
}

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
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
  LayerParameter layer_param;
  CuDNNSoftmaxLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

#endif


}  // namespace caffe
