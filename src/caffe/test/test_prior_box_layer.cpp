#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/prior_box_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class PriorBoxLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  PriorBoxLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 10, 10, 10)),
        blob_data_(new Blob<Dtype>(10, 3, 100, 100)),
        blob_top_(new Blob<Dtype>()),
        min_size_(4),
        max_size_(9) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_data_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_data_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PriorBoxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_data_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  size_t min_size_;
  size_t max_size_;
};

TYPED_TEST_CASE(PriorBoxLayerTest, TestDtypes);

TYPED_TEST(PriorBoxLayerTest, TestSetup) {
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_max_size(this->max_size_);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 100 * 2 * 4);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(PriorBoxLayerTest, TestSetupMultiSize) {
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_min_size(this->min_size_ + 10);
  prior_box_param->add_max_size(this->max_size_);
  prior_box_param->add_max_size(this->max_size_ + 10);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 100 * 4 * 4);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(PriorBoxLayerTest, TestSetupNoMaxSize) {
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 100 * 1 * 4);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(PriorBoxLayerTest, TestSetupMultiSizeNoMaxSize) {
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_min_size(this->min_size_ + 10);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 100 * 2 * 4);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(PriorBoxLayerTest, TestSetupAspectRatio1) {
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_max_size(this->max_size_);
  prior_box_param->add_aspect_ratio(1.);
  prior_box_param->add_aspect_ratio(2.);
  prior_box_param->set_flip(false);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 100 * 3 * 4);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(PriorBoxLayerTest, TestSetupAspectRatioNoFlip) {
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_max_size(this->max_size_);
  prior_box_param->add_aspect_ratio(2.);
  prior_box_param->add_aspect_ratio(3.);
  prior_box_param->set_flip(false);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 100 * 4 * 4);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(PriorBoxLayerTest, TestSetupAspectRatio) {
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_max_size(this->max_size_);
  prior_box_param->add_aspect_ratio(2.);
  prior_box_param->add_aspect_ratio(3.);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 100 * 6 * 4);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(PriorBoxLayerTest, TestSetupAspectRatioMultiSize) {
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_min_size(this->min_size_ + 10);
  prior_box_param->add_max_size(this->max_size_);
  prior_box_param->add_max_size(this->max_size_ + 10);
  prior_box_param->add_aspect_ratio(2.);
  prior_box_param->add_aspect_ratio(3.);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 100 * 12 * 4);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(PriorBoxLayerTest, TestCPU) {
  const TypeParam eps = 1e-6;
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_max_size(this->max_size_);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int dim = this->blob_top_->height();
  // pick a few generated priors and compare against the expected number.
  // first prior
  EXPECT_NEAR(top_data[0], 0.03, eps);
  EXPECT_NEAR(top_data[1], 0.03, eps);
  EXPECT_NEAR(top_data[2], 0.07, eps);
  EXPECT_NEAR(top_data[3], 0.07, eps);
  // second prior
  EXPECT_NEAR(top_data[4], 0.02, eps);
  EXPECT_NEAR(top_data[5], 0.02, eps);
  EXPECT_NEAR(top_data[6], 0.08, eps);
  EXPECT_NEAR(top_data[7], 0.08, eps);
  // prior in the 5-th row and 5-th col
  EXPECT_NEAR(top_data[4*10*2*4+4*2*4], 0.43, eps);
  EXPECT_NEAR(top_data[4*10*2*4+4*2*4+1], 0.43, eps);
  EXPECT_NEAR(top_data[4*10*2*4+4*2*4+2], 0.47, eps);
  EXPECT_NEAR(top_data[4*10*2*4+4*2*4+3], 0.47, eps);

  // check variance
  top_data += dim;
  for (int d = 0; d < dim; ++d) {
    EXPECT_NEAR(top_data[d], 0.1, eps);
  }
}

TYPED_TEST(PriorBoxLayerTest, TestCPUNoMaxSize) {
  const TypeParam eps = 1e-6;
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int dim = this->blob_top_->height();
  // pick a few generated priors and compare against the expected number.
  // first prior
  EXPECT_NEAR(top_data[0], 0.03, eps);
  EXPECT_NEAR(top_data[1], 0.03, eps);
  EXPECT_NEAR(top_data[2], 0.07, eps);
  EXPECT_NEAR(top_data[3], 0.07, eps);
  // prior in the 5-th row and 5-th col
  EXPECT_NEAR(top_data[4*10*1*4+4*1*4], 0.43, eps);
  EXPECT_NEAR(top_data[4*10*1*4+4*1*4+1], 0.43, eps);
  EXPECT_NEAR(top_data[4*10*1*4+4*1*4+2], 0.47, eps);
  EXPECT_NEAR(top_data[4*10*1*4+4*1*4+3], 0.47, eps);

  // check variance
  top_data += dim;
  for (int d = 0; d < dim; ++d) {
    EXPECT_NEAR(top_data[d], 0.1, eps);
  }
}

TYPED_TEST(PriorBoxLayerTest, TestCPUVariance1) {
  const TypeParam eps = 1e-6;
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_max_size(this->max_size_);
  prior_box_param->add_variance(1.);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int dim = this->blob_top_->height();
  // pick a few generated priors and compare against the expected number.
  // first prior
  EXPECT_NEAR(top_data[0], 0.03, eps);
  EXPECT_NEAR(top_data[1], 0.03, eps);
  EXPECT_NEAR(top_data[2], 0.07, eps);
  EXPECT_NEAR(top_data[3], 0.07, eps);
  // second prior
  EXPECT_NEAR(top_data[4], 0.02, eps);
  EXPECT_NEAR(top_data[5], 0.02, eps);
  EXPECT_NEAR(top_data[6], 0.08, eps);
  EXPECT_NEAR(top_data[7], 0.08, eps);
  // prior in the 5-th row and 5-th col
  EXPECT_NEAR(top_data[4*10*2*4+4*2*4], 0.43, eps);
  EXPECT_NEAR(top_data[4*10*2*4+4*2*4+1], 0.43, eps);
  EXPECT_NEAR(top_data[4*10*2*4+4*2*4+2], 0.47, eps);
  EXPECT_NEAR(top_data[4*10*2*4+4*2*4+3], 0.47, eps);

  // check variance
  top_data += dim;
  for (int d = 0; d < dim; ++d) {
    EXPECT_NEAR(top_data[d], 1., eps);
  }
}

TYPED_TEST(PriorBoxLayerTest, TestCPUVarianceMulti) {
  const TypeParam eps = 1e-6;
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_max_size(this->max_size_);
  prior_box_param->add_variance(0.1);
  prior_box_param->add_variance(0.2);
  prior_box_param->add_variance(0.3);
  prior_box_param->add_variance(0.4);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int dim = this->blob_top_->height();
  // pick a few generated priors and compare against the expected number.
  // first prior
  EXPECT_NEAR(top_data[0], 0.03, eps);
  EXPECT_NEAR(top_data[1], 0.03, eps);
  EXPECT_NEAR(top_data[2], 0.07, eps);
  EXPECT_NEAR(top_data[3], 0.07, eps);
  // second prior
  EXPECT_NEAR(top_data[4], 0.02, eps);
  EXPECT_NEAR(top_data[5], 0.02, eps);
  EXPECT_NEAR(top_data[6], 0.08, eps);
  EXPECT_NEAR(top_data[7], 0.08, eps);
  // prior in the 5-th row and 5-th col
  EXPECT_NEAR(top_data[4*10*2*4+4*2*4], 0.43, eps);
  EXPECT_NEAR(top_data[4*10*2*4+4*2*4+1], 0.43, eps);
  EXPECT_NEAR(top_data[4*10*2*4+4*2*4+2], 0.47, eps);
  EXPECT_NEAR(top_data[4*10*2*4+4*2*4+3], 0.47, eps);

  // check variance
  top_data += dim;
  for (int d = 0; d < dim; ++d) {
    EXPECT_NEAR(top_data[d], 0.1 * (d % 4 + 1), eps);
  }
}

TYPED_TEST(PriorBoxLayerTest, TestCPUAspectRatioNoFlip) {
  const TypeParam eps = 1e-6;
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_max_size(this->max_size_);
  prior_box_param->add_aspect_ratio(2.);
  prior_box_param->set_flip(false);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int dim = this->blob_top_->height();
  // pick a few generated priors and compare against the expected number.
  // first prior
  EXPECT_NEAR(top_data[0], 0.03, eps);
  EXPECT_NEAR(top_data[1], 0.03, eps);
  EXPECT_NEAR(top_data[2], 0.07, eps);
  EXPECT_NEAR(top_data[3], 0.07, eps);
  // second prior
  EXPECT_NEAR(top_data[4], 0.02, eps);
  EXPECT_NEAR(top_data[5], 0.02, eps);
  EXPECT_NEAR(top_data[6], 0.08, eps);
  EXPECT_NEAR(top_data[7], 0.08, eps);
  // third prior
  EXPECT_NEAR(top_data[8], 0.05 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[9], 0.05 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[10], 0.05 + 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[11], 0.05 + 0.01*sqrt(2.), eps);
  // prior in the 5-th row and 5-th col
  EXPECT_NEAR(top_data[4*10*3*4+4*3*4], 0.43, eps);
  EXPECT_NEAR(top_data[4*10*3*4+4*3*4+1], 0.43, eps);
  EXPECT_NEAR(top_data[4*10*3*4+4*3*4+2], 0.47, eps);
  EXPECT_NEAR(top_data[4*10*3*4+4*3*4+3], 0.47, eps);
  // prior with ratio 1:2 in the 5-th row and 5-th col
  EXPECT_NEAR(top_data[4*10*3*4+4*3*4+8], 0.45 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[4*10*3*4+4*3*4+9], 0.45 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[4*10*3*4+4*3*4+10], 0.45 + 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[4*10*3*4+4*3*4+11], 0.45 + 0.01*sqrt(2.), eps);

  // check variance
  top_data += dim;
  for (int d = 0; d < dim; ++d) {
    EXPECT_NEAR(top_data[d], 0.1, eps);
  }
}

TYPED_TEST(PriorBoxLayerTest, TestCPUAspectRatio) {
  const TypeParam eps = 1e-6;
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_max_size(this->max_size_);
  prior_box_param->add_aspect_ratio(2.);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int dim = this->blob_top_->height();
  // pick a few generated priors and compare against the expected number.
  // first prior
  EXPECT_NEAR(top_data[0], 0.03, eps);
  EXPECT_NEAR(top_data[1], 0.03, eps);
  EXPECT_NEAR(top_data[2], 0.07, eps);
  EXPECT_NEAR(top_data[3], 0.07, eps);
  // second prior
  EXPECT_NEAR(top_data[4], 0.02, eps);
  EXPECT_NEAR(top_data[5], 0.02, eps);
  EXPECT_NEAR(top_data[6], 0.08, eps);
  EXPECT_NEAR(top_data[7], 0.08, eps);
  // third prior
  EXPECT_NEAR(top_data[8], 0.05 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[9], 0.05 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[10], 0.05 + 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[11], 0.05 + 0.01*sqrt(2.), eps);
  // forth prior
  EXPECT_NEAR(top_data[12], 0.05 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[13], 0.05 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[14], 0.05 + 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[15], 0.05 + 0.02*sqrt(2.), eps);
  // prior in the 5-th row and 5-th col
  EXPECT_NEAR(top_data[4*10*4*4+4*4*4], 0.43, eps);
  EXPECT_NEAR(top_data[4*10*4*4+4*4*4+1], 0.43, eps);
  EXPECT_NEAR(top_data[4*10*4*4+4*4*4+2], 0.47, eps);
  EXPECT_NEAR(top_data[4*10*4*4+4*4*4+3], 0.47, eps);
  // prior with ratio 1:2 in the 5-th row and 5-th col
  EXPECT_NEAR(top_data[4*10*4*4+4*4*4+8], 0.45 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[4*10*4*4+4*4*4+9], 0.45 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[4*10*4*4+4*4*4+10], 0.45 + 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[4*10*4*4+4*4*4+11], 0.45 + 0.01*sqrt(2.), eps);
  // prior with ratio 2:1 in the 5-th row and 5-th col
  EXPECT_NEAR(top_data[4*10*4*4+4*4*4+12], 0.45 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[4*10*4*4+4*4*4+13], 0.45 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[4*10*4*4+4*4*4+14], 0.45 + 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[4*10*4*4+4*4*4+15], 0.45 + 0.02*sqrt(2.), eps);

  // check variance
  top_data += dim;
  for (int d = 0; d < dim; ++d) {
    EXPECT_NEAR(top_data[d], 0.1, eps);
  }
}

TYPED_TEST(PriorBoxLayerTest, TestCPUAspectRatioMultiSize) {
  const TypeParam eps = 1e-6;
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_min_size(this->min_size_ + 4);
  prior_box_param->add_max_size(this->max_size_);
  prior_box_param->add_max_size(this->max_size_ + 9);
  prior_box_param->add_aspect_ratio(2.);
  prior_box_param->set_clip(true);
  PriorBoxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int dim = this->blob_top_->height();
  // pick a few generated priors and compare against the expected number.
  // first prior
  EXPECT_NEAR(top_data[0], 0.03, eps);
  EXPECT_NEAR(top_data[1], 0.03, eps);
  EXPECT_NEAR(top_data[2], 0.07, eps);
  EXPECT_NEAR(top_data[3], 0.07, eps);
  // second prior
  EXPECT_NEAR(top_data[4], 0.02, eps);
  EXPECT_NEAR(top_data[5], 0.02, eps);
  EXPECT_NEAR(top_data[6], 0.08, eps);
  EXPECT_NEAR(top_data[7], 0.08, eps);
  // third prior
  EXPECT_NEAR(top_data[8], 0.05 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[9], 0.05 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[10], 0.05 + 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[11], 0.05 + 0.01*sqrt(2.), eps);
  // forth prior
  EXPECT_NEAR(top_data[12], 0.05 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[13], 0.05 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[14], 0.05 + 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[15], 0.05 + 0.02*sqrt(2.), eps);
  // fifth prior
  EXPECT_NEAR(top_data[16], 0.01, eps);
  EXPECT_NEAR(top_data[17], 0.01, eps);
  EXPECT_NEAR(top_data[18], 0.09, eps);
  EXPECT_NEAR(top_data[19], 0.09, eps);
  // sixth prior
  EXPECT_NEAR(top_data[20], 0.00, eps);
  EXPECT_NEAR(top_data[21], 0.00, eps);
  EXPECT_NEAR(top_data[22], 0.11, eps);
  EXPECT_NEAR(top_data[23], 0.11, eps);
  // seventh prior
  EXPECT_NEAR(top_data[24], 0.00, eps);
  EXPECT_NEAR(top_data[25], 0.05 - 0.04/sqrt(2.), eps);
  EXPECT_NEAR(top_data[26], 0.05 + 0.04*sqrt(2.), eps);
  EXPECT_NEAR(top_data[27], 0.05 + 0.04/sqrt(2.), eps);
  // forth prior
  EXPECT_NEAR(top_data[28], 0.05 - 0.04/sqrt(2.), eps);
  EXPECT_NEAR(top_data[29], 0.00, eps);
  EXPECT_NEAR(top_data[30], 0.05 + 0.04/sqrt(2.), eps);
  EXPECT_NEAR(top_data[31], 0.05 + 0.04*sqrt(2.), eps);
  // prior in the 5-th row and 5-th col
  EXPECT_NEAR(top_data[8*10*4*4+8*4*4], 0.43, eps);
  EXPECT_NEAR(top_data[8*10*4*4+8*4*4+1], 0.43, eps);
  EXPECT_NEAR(top_data[8*10*4*4+8*4*4+2], 0.47, eps);
  EXPECT_NEAR(top_data[8*10*4*4+8*4*4+3], 0.47, eps);
  // prior with ratio 1:2 in the 5-th row and 5-th col
  EXPECT_NEAR(top_data[8*10*4*4+8*4*4+8], 0.45 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[8*10*4*4+8*4*4+9], 0.45 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[8*10*4*4+8*4*4+10], 0.45 + 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[8*10*4*4+8*4*4+11], 0.45 + 0.01*sqrt(2.), eps);
  // prior with ratio 2:1 in the 5-th row and 5-th col
  EXPECT_NEAR(top_data[8*10*4*4+8*4*4+12], 0.45 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[8*10*4*4+8*4*4+13], 0.45 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[8*10*4*4+8*4*4+14], 0.45 + 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[8*10*4*4+8*4*4+15], 0.45 + 0.02*sqrt(2.), eps);

  // check variance
  top_data += dim;
  for (int d = 0; d < dim; ++d) {
    EXPECT_NEAR(top_data[d], 0.1, eps);
  }
}

TYPED_TEST(PriorBoxLayerTest, TestCPUFixStep) {
  const TypeParam eps = 1e-6;
  LayerParameter layer_param;
  PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
  prior_box_param->add_min_size(this->min_size_);
  prior_box_param->add_max_size(this->max_size_);
  prior_box_param->add_aspect_ratio(2.);
  prior_box_param->set_img_size(100);
  prior_box_param->set_step(10);
  PriorBoxLayer<TypeParam> layer(layer_param);
  vector<int> shape(4, 10);
  shape[2] = 20;
  this->blob_bottom_->Reshape(shape);
  shape[1] = 3;
  shape[2] = 200;
  shape[3] = 100;
  this->blob_data_->Reshape(shape);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int dim = this->blob_top_->height();
  // pick a few generated priors and compare against the expected number.
  // first prior
  EXPECT_NEAR(top_data[0], 0.03, eps);
  EXPECT_NEAR(top_data[1], 0.03, eps);
  EXPECT_NEAR(top_data[2], 0.07, eps);
  EXPECT_NEAR(top_data[3], 0.07, eps);
  // second prior
  EXPECT_NEAR(top_data[4], 0.02, eps);
  EXPECT_NEAR(top_data[5], 0.02, eps);
  EXPECT_NEAR(top_data[6], 0.08, eps);
  EXPECT_NEAR(top_data[7], 0.08, eps);
  // third prior
  EXPECT_NEAR(top_data[8], 0.05 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[9], 0.05 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[10], 0.05 + 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[11], 0.05 + 0.01*sqrt(2.), eps);
  // forth prior
  EXPECT_NEAR(top_data[12], 0.05 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[13], 0.05 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[14], 0.05 + 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[15], 0.05 + 0.02*sqrt(2.), eps);
  // prior in the 15-th row and 5-th col
  EXPECT_NEAR(top_data[14*10*4*4+4*4*4], 0.43, eps);
  EXPECT_NEAR(top_data[14*10*4*4+4*4*4+1], 1.43, eps);
  EXPECT_NEAR(top_data[14*10*4*4+4*4*4+2], 0.47, eps);
  EXPECT_NEAR(top_data[14*10*4*4+4*4*4+3], 1.47, eps);
  // prior with ratio 1:2 in the 15-th row and 5-th col
  EXPECT_NEAR(top_data[14*10*4*4+4*4*4+8], 0.45 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[14*10*4*4+4*4*4+9], 1.45 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[14*10*4*4+4*4*4+10], 0.45 + 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[14*10*4*4+4*4*4+11], 1.45 + 0.01*sqrt(2.), eps);
  // prior with ratio 2:1 in the 15-th row and 5-th col
  EXPECT_NEAR(top_data[14*10*4*4+4*4*4+12], 0.45 - 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[14*10*4*4+4*4*4+13], 1.45 - 0.02*sqrt(2.), eps);
  EXPECT_NEAR(top_data[14*10*4*4+4*4*4+14], 0.45 + 0.01*sqrt(2.), eps);
  EXPECT_NEAR(top_data[14*10*4*4+4*4*4+15], 1.45 + 0.02*sqrt(2.), eps);

  // check variance
  top_data += dim;
  for (int d = 0; d < dim; ++d) {
    EXPECT_NEAR(top_data[d], 0.1, eps);
  }
}

}  // namespace caffe
