#include <algorithm>
#include <cmath>
#include <cstdlib>
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
class TripletLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TripletLossLayerTest()
      : blob_bottom_data_i_(new Blob<Dtype>(512, 1, 1, 1)),
        blob_bottom_data_j_(new Blob<Dtype>(512, 1, 1, 1)),
	blob_bottom_data_k_(new Blob<Dtype>(512, 1, 1, 1)),
	blob_bottom_data_l_(new Blob<Dtype>(512, 1, 1, 1)),
	blob_bottom_data_m_(new Blob<Dtype>(512, 1, 1, 1)),	
        blob_bottom_y_(new Blob<Dtype>(512, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_i_);
    blob_bottom_vec_.push_back(blob_bottom_data_i_);
    filler.Fill(this->blob_bottom_data_j_);
    blob_bottom_vec_.push_back(blob_bottom_data_j_);
    filler.Fill(this->blob_bottom_data_k_);
    blob_bottom_vec_.push_back(blob_bottom_data_k_);
    filler.Fill(this->blob_bottom_data_l_);
    blob_bottom_vec_.push_back(blob_bottom_data_l_);
    filler.Fill(this->blob_bottom_data_m_);
    blob_bottom_vec_.push_back(blob_bottom_data_m_);
    for (int i = 0; i < blob_bottom_y_->count(); ++i) {
      blob_bottom_y_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;  // 0 or 1
    }
    blob_bottom_vec_.push_back(blob_bottom_y_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~TripletLossLayerTest() {
    delete blob_bottom_data_i_;
    delete blob_bottom_data_j_;
    delete blob_bottom_data_k_;
    delete blob_bottom_data_l_;
    delete blob_bottom_data_m_;
    delete blob_bottom_y_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_i_;
  Blob<Dtype>* const blob_bottom_data_j_;
  Blob<Dtype>* const blob_bottom_data_k_;
  Blob<Dtype>* const blob_bottom_data_l_;
  Blob<Dtype>* const blob_bottom_data_m_;
  Blob<Dtype>* const blob_bottom_y_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TripletLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(TripletLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TripletLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const Dtype margin = layer_param.triplet_loss_param().margin();
  const Dtype losstype = layer_param.triplet_loss_param().losstype();
  const int num = this->blob_bottom_data_i_->num();
  const int channels = this->blob_bottom_data_i_->channels();
  Dtype loss(0);
  if (losstype == 0) {
  for (int i = 0; i < num; ++i) {
    Dtype dist_sq(0);
    Dtype dist_par(0);
    for (int j = 0; j < channels; ++j) {
      Dtype diff_pos = this->blob_bottom_data_i_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_j_->cpu_data()[i*channels+j];
      dist_sq += diff_pos*diff_pos;
      Dtype diff_neg = this->blob_bottom_data_i_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_k_->cpu_data()[i*channels+j];
      dist_sq -= diff_neg*diff_neg;
      Dtype diff_par = this->blob_bottom_data_l_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_m_->cpu_data()[i*channels+j];
      dist_par = diff_par*diff_par;
    }
    loss += std::max(margin + dist_sq, Dtype(0.0));
    loss += dist_par;
  }
  } else {
  for (int i = 0; i < num; ++i) {
    Dtype dist_sq(0);
    Dtype dist_par(0);
    for (int j = 0; j < channels; ++j) {
      Dtype diff_pos = this->blob_bottom_data_i_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_j_->cpu_data()[i*channels+j];
      dist_sq += diff_pos*diff_pos;
      dist_sq += margin;
      Dtype diff_neg = this->blob_bottom_data_i_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_k_->cpu_data()[i*channels+j];
      dist_sq = 1 - diff_neg*diff_neg/dist_sq;
      Dtype diff_par = this->blob_bottom_data_l_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_m_->cpu_data()[i*channels+j];
      dist_par = diff_par*diff_par;
    }
    loss += std::max(dist_sq, Dtype(0.0));
    loss += dist_par;
  }
  }
  loss /= static_cast<Dtype>(num) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
}

TYPED_TEST(TripletLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TripletLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the first 5 bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 4);
}
}  // namespace caffe
