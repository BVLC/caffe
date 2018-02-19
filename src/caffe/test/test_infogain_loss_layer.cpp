#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/infogain_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class InfogainLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  InfogainLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(4, 2, 5, 2)),
        blob_bottom_label_(new Blob<Dtype>(4, 2, 1, 2)),
        blob_bottom_infogain_(new Blob<Dtype>(1, 1, 5, 5)),
        blob_top_loss_(new Blob<Dtype>()),
        blob_top_prob_(new Blob<Dtype>()),
        inner_(2), outer_(4*2), num_labels_(5) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_min(-0.5);
    filler_param.set_max(2.0);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] =
        caffe_rng_rand() % num_labels_;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    filler_param.set_min(0.1);
    filler_param.set_max(2.0);
    UniformFiller<Dtype> infogain_filler(filler_param);
    infogain_filler.Fill(this->blob_bottom_infogain_);
    blob_bottom_vec_.push_back(blob_bottom_infogain_);
    blob_top_vec_.push_back(blob_top_loss_);
    blob_top_vec_.push_back(blob_top_prob_);
  }
  virtual ~InfogainLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_infogain_;
    delete blob_top_loss_;
    delete blob_top_prob_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_infogain_;
  Blob<Dtype>* const blob_top_loss_;
  Blob<Dtype>* const blob_top_prob_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int inner_, outer_, num_labels_;
};

TYPED_TEST_CASE(InfogainLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(InfogainLossLayerTest, TestInfogainLoss) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_infogain_loss_param()->set_axis(2);
  layer_param.clear_loss_weight();
  layer_param.add_loss_weight(1);
  layer_param.add_loss_weight(0);
  /*vector<float>* lw = layer_param.mutable_loss_weight();
  lw->clear();
  lw->push_back(1);
  lw->push_back(1);*/
  InfogainLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* data = this->blob_bottom_vec_[0]->cpu_data();
  const Dtype* prob = this->blob_top_vec_[1]->cpu_data();
  const Dtype* labels = this->blob_bottom_vec_[1]->cpu_data();
  const Dtype* H = this->blob_bottom_vec_[2]->cpu_data();
  // first. test the prob top
  CHECK_EQ(this->blob_bottom_vec_[0]->num_axes(),
    this->blob_top_vec_[1]->num_axes())
      << "prob top shape not match bottom data";
  for (int ai = 0 ; ai < this->blob_bottom_vec_[0]->num_axes(); ai++) {
    CHECK_EQ(this->blob_bottom_vec_[0]->shape(ai),
      this->blob_top_vec_[1]->shape(ai))
        << "prob top shape not match bottom data";
  }
  vector<Dtype> est_prob(this->num_labels_, 0);
  for ( int i = 0 ; i < this->outer_; i++ ) {
    for ( int j = 0; j < this->inner_; j++ ) {
      Dtype den = 0;
      for ( int  l = 0; l < this->num_labels_; l++ ) {
        est_prob[l] = std::exp(
          data[i*this->num_labels_*this->inner_ + l*this->inner_ + j]);
        den += est_prob[l];
      }
      for ( int l = 0; l < this->num_labels_; l++ ) {
        EXPECT_NEAR(prob[i*this->num_labels_*this->inner_ + l*this->inner_ + j],
          est_prob[l]/den, 1e-6);
      }
    }
  }
  Dtype loss = 0;  // loss from prob top
  for ( int i = 0 ; i < this->outer_; i++ ) {
    for ( int j = 0; j < this->inner_; j++ ) {
      int gt = static_cast<int>(labels[i*this->inner_+j]);
      for ( int l = 0; l < this->num_labels_; l++ ) {
        loss -= H[gt*this->num_labels_ + l] *
          log(std::max(
            prob[i*this->num_labels_*this->inner_ + l*this->inner_ + j],
            Dtype(kLOG_THRESHOLD)));
      }
    }
  }
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0],
    loss/(this->outer_*this->inner_), 1e-6);
}

TYPED_TEST(InfogainLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_infogain_loss_param()->set_axis(2);
  InfogainLossLayer<Dtype> layer(layer_param);
  this->blob_top_vec_.clear();  // ignore prob top.
  this->blob_top_vec_.push_back(this->blob_top_loss_);
  GradientChecker<Dtype> checker(1e-4, 2e-2, 1701);  // no "kink"
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
