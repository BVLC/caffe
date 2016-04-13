#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/cudnn_batch_norm_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 2
#define INPUT_DATA_SIZE 3

namespace caffe {

  template <typename TypeParam>
  class BatchNormLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
   protected:
    BatchNormLayerTest()
        : blob_bottom_(new Blob<Dtype>(5, 2, 3, 4)),
          blob_top_(new Blob<Dtype>()) {
      // fill the values
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~BatchNormLayerTest() { delete blob_bottom_; delete blob_top_; }
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
  };

  TYPED_TEST_CASE(BatchNormLayerTest, TestDtypesAndDevices);

  TYPED_TEST(BatchNormLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    BatchNormParameter* bn_param = layer_param.mutable_batch_norm_param();
    FillerParameter *scale_param = bn_param->mutable_scale_filler();
    scale_param->set_value(1);
    FillerParameter *bias_param = bn_param->mutable_bias_filler();
    bias_param->set_value(0);

    bn_param->set_eps(0.);

    BatchNormLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Test mean
    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();

    for (int j = 0; j < channels; ++j) {
      Dtype sum = 0, var = 0;
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = this->blob_top_->data_at(i, j, k, l);
            sum += data;
            var += data * data;
          }
        }
      }
      sum /= height * width * num;
      var /= height * width * num;

      const Dtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
    }
  }

  TYPED_TEST(BatchNormLayerTest, TestForwardInplace) {
    typedef typename TypeParam::Dtype Dtype;
    Blob<Dtype> blob_inplace(5, 2, 3, 4);
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    LayerParameter layer_param;
    BatchNormParameter* bn_param = layer_param.mutable_batch_norm_param();
    FillerParameter *scale_param = bn_param->mutable_scale_filler();
    scale_param->set_value(1);
    FillerParameter *bias_param = bn_param->mutable_bias_filler();
    bias_param->set_value(0);

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_inplace);
    blob_bottom_vec.push_back(&blob_inplace);
    blob_top_vec.push_back(&blob_inplace);

    BatchNormLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec, blob_top_vec);
    layer.Forward(blob_bottom_vec, blob_top_vec);

    // Test mean
    int num = blob_inplace.num();
    int channels = blob_inplace.channels();
    int height = blob_inplace.height();
    int width = blob_inplace.width();

    for (int j = 0; j < channels; ++j) {
      Dtype sum = 0, var = 0;
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = blob_inplace.data_at(i, j, k, l);
            sum += data;
            var += data * data;
          }
        }
      }
      sum /= height * width * num;
      var /= height * width * num;

      const Dtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
    }
  }

  TYPED_TEST(BatchNormLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    BatchNormLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-4);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNBatchNormLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  CuDNNBatchNormLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_mean(-10);
    filler_param.set_std(5);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNBatchNormLayerTest() { delete blob_bottom_; delete blob_top_; }
  void checkMeanVar(const Blob<Dtype> *blob_bottom, int num,
    int channels, int height, int width);
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void CuDNNBatchNormLayerTest<TypeParam>::checkMeanVar(
    const Blob<TypeParam> *top,
    int num, int channels, int height, int width) {
  typedef TypeParam Dtype;

  for (int j = 0; j < channels; ++j) {
    Dtype mean = 0, var = 0;
    for (int i = 0; i < num; ++i) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = top->data_at(i, j, k, l);
          mean += data;
          var += data * data;
        }
      }
    }
    mean /= num * height * width;
    var /= num * height * width;

    const Dtype kErrorBound = 0.001;
    EXPECT_NEAR(0, mean, kErrorBound);
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST_CASE(CuDNNBatchNormLayerTest, TestDtypes);

TYPED_TEST(CuDNNBatchNormLayerTest, TestForward) {
  Caffe::set_random_seed(1701);
  typedef TypeParam Dtype;
  LayerParameter layer_param;
  BatchNormParameter* bn_param = layer_param.mutable_batch_norm_param();
  FillerParameter *scale_param = bn_param->mutable_scale_filler();
  scale_param->set_value(1);
  bn_param->set_eps(0.);

  CuDNNBatchNormLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  Dtype mean, var;
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  this->checkMeanVar(this->blob_top_, num, channels, height, width);
}

TYPED_TEST(CuDNNBatchNormLayerTest, TestGradient) {
  typedef TypeParam Dtype;
  LayerParameter layer_param;
  BatchNormParameter* bn_param = layer_param.mutable_batch_norm_param();
  FillerParameter *scale_param = bn_param->mutable_scale_filler();
  scale_param->set_value(1);
  FillerParameter *bias_param = bn_param->mutable_bias_filler();
  bias_param->set_value(0);

  CuDNNBatchNormLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 4e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
#endif

}  // namespace caffe
