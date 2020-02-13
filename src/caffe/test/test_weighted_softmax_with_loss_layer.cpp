#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/weighted_softmax_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class WeightedSoftmaxWithLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  WeightedSoftmaxWithLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 2, 3)),
        blob_bottom_weight_(new Blob<Dtype>(10, 1, 2, 3)),
        blob_top_loss_(new Blob<Dtype>()),
        blob_top_prob_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    filler_param.set_min(0.1);
    filler_param.set_max(1);
    UniformFiller<Dtype> ufiller(filler_param);
    ufiller.Fill(this->blob_bottom_weight_);
    blob_bottom_vec_.push_back(blob_bottom_weight_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~WeightedSoftmaxWithLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_weight_;
    delete blob_top_loss_;
    delete blob_top_prob_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_weight_;
  Blob<Dtype>* const blob_top_loss_;
  Blob<Dtype>* const blob_top_prob_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(WeightedSoftmaxWithLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(WeightedSoftmaxWithLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  // only loss top
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->blob_top_loss_);
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  WeightedSoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WeightedSoftmaxWithLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  // top loss + prob
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->blob_top_loss_);
  this->blob_top_vec_.push_back(this->blob_top_prob_);

  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  layer_param.add_loss_weight(0);
  // no normalization
  layer_param.mutable_loss_param()->
    set_normalization(LossParameter_NormalizationMode_NONE);
  WeightedSoftmaxWithLossLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype layer_loss = this->blob_top_loss_->cpu_data()[0];

  Dtype mloss = 0;
  Dtype agg_weight = 0;
  vector<int> prob_ind(4, 0);
  vector<int> label_ind(4, 0);
  for ( prob_ind[0]=0;
        prob_ind[0] < this->blob_top_prob_->shape(0);
        prob_ind[0]++ ) {
    for ( prob_ind[2]=0;
          prob_ind[2] < this->blob_top_prob_->shape(2);
          prob_ind[2]++ ) {
      for ( prob_ind[3]=0;
            prob_ind[3] < this->blob_top_prob_->shape(3);
            prob_ind[3]++ ) {
        label_ind = prob_ind;
        label_ind[1] = 0;
        const int label_value = static_cast<int>
          (this->blob_bottom_label_->data_at(label_ind));
        prob_ind[1] = label_value;
        mloss -= this->blob_bottom_weight_->data_at(label_ind)
         * log(std::max(this->blob_top_prob_->data_at(prob_ind),
                        Dtype(FLT_MIN)));
        agg_weight += this->blob_bottom_weight_->data_at(label_ind);
      }
    }
  }
  EXPECT_NEAR(layer_loss, mloss, 1e-4);
}

TYPED_TEST(WeightedSoftmaxWithLossLayerTest, TestForwardIgnoreLabelOnce) {
  typedef typename TypeParam::Dtype Dtype;
  // top loss + prob
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->blob_top_loss_);
  this->blob_top_vec_.push_back(this->blob_top_prob_);

  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  layer_param.add_loss_weight(0);
  // VALID normalization
  layer_param.mutable_loss_param()->
    set_normalization(LossParameter_NormalizationMode_VALID);
  const int ignore_label = 0;
  layer_param.mutable_loss_param()->set_ignore_label(ignore_label);
  WeightedSoftmaxWithLossLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype layer_loss = this->blob_top_loss_->cpu_data()[0];

  Dtype mloss = 0;
  Dtype agg_weight = 0;
  vector<int> prob_ind(4, 0);
  vector<int> label_ind(4, 0);
  for ( prob_ind[0]=0;
        prob_ind[0] < this->blob_top_prob_->shape(0);
        prob_ind[0]++ ) {
    for ( prob_ind[2]=0;
          prob_ind[2] < this->blob_top_prob_->shape(2);
          prob_ind[2]++ ) {
      for ( prob_ind[3]=0;
            prob_ind[3] < this->blob_top_prob_->shape(3);
            prob_ind[3]++ ) {
        label_ind = prob_ind;
        label_ind[1] = 0;
        const int label_value = static_cast<int>
          (this->blob_bottom_label_->data_at(label_ind));
        if (label_value == ignore_label)
          continue;
        prob_ind[1] = label_value;
        mloss -= this->blob_bottom_weight_->data_at(label_ind)
         * log(std::max(this->blob_top_prob_->data_at(prob_ind),
                        Dtype(FLT_MIN)));
        agg_weight += this->blob_bottom_weight_->data_at(label_ind);
      }
    }
  }
  EXPECT_NEAR(layer_loss, mloss/std::max(Dtype(1.0), agg_weight), 1e-4);
}

TYPED_TEST(WeightedSoftmaxWithLossLayerTest, TestForwardIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  // only loss top
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->blob_top_loss_);

  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  // First, compute the loss with all labels
  scoped_ptr<WeightedSoftmaxWithLossLayer<Dtype> > layer(
      new WeightedSoftmaxWithLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
  // Now, accumulate the loss, ignoring each label in {0, ..., 4} in turn.
  Dtype accum_loss = 0;
  for (int label = 0; label < 5; ++label) {
    layer_param.mutable_loss_param()->set_ignore_label(label);
    layer.reset(new WeightedSoftmaxWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    accum_loss += this->blob_top_loss_->cpu_data()[0];
  }
  // Check that each label was included all but once.
  EXPECT_NEAR(4 * full_loss, accum_loss, 1e-4);
}

TYPED_TEST(WeightedSoftmaxWithLossLayerTest, TestGradientIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  // only loss top
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->blob_top_loss_);

  LayerParameter layer_param;
  // labels are in {0, ..., 4}, so we'll ignore about a fifth of them
  layer_param.mutable_loss_param()->set_ignore_label(0);
  WeightedSoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WeightedSoftmaxWithLossLayerTest, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  // only loss top
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->blob_top_loss_);

  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  WeightedSoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
