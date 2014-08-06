#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class AccuracyLayerTest : public ::testing::Test {
 protected:
  AccuracyLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(100, 10, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(100, 1, 1, 1)),
        blob_top_(new Blob<Dtype>()),
        top_k_(3) {
    // fill the probability values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    shared_ptr<Caffe::RNG> rng(new Caffe::RNG(prefetch_rng_seed));
    caffe::rng_t* prefetch_rng =
          static_cast<caffe::rng_t*>(rng->generator());
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < 100; ++i) {
      label_data[i] = (*prefetch_rng)() % 10;
    }

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~AccuracyLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int top_k_;
};

TYPED_TEST_CASE(AccuracyLayerTest, TestDtypes);

TYPED_TEST(AccuracyLayerTest, TestSetup) {
  LayerParameter layer_param;
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(AccuracyLayerTest, TestSetupTopK) {
  LayerParameter layer_param;
  AccuracyParameter* accuracy_param =
      layer_param.mutable_accuracy_param();
  accuracy_param->set_top_k(5);
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(AccuracyLayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  TypeParam max_value;
  int max_id;
  int num_correct_labels = 0;
  for (int i = 0; i < 100; ++i) {
    max_value = -FLT_MAX;
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    if (max_id == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      ++num_correct_labels;
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / 100.0, 1e-4);
}

TYPED_TEST(AccuracyLayerTest, TestForwardCPUTopK) {
  LayerParameter layer_param;
  AccuracyParameter* accuracy_param = layer_param.mutable_accuracy_param();
  accuracy_param->set_top_k(this->top_k_);
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  TypeParam current_value;
  int current_rank;
  int num_correct_labels = 0;
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 10; ++j) {
      current_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
      current_rank = 0;
      for (int k = 0; k < 10; ++k) {
        if (this->blob_bottom_data_->data_at(i, k, 0, 0) > current_value) {
          ++current_rank;
        }
      }
      if (current_rank < this->top_k_ &&
          j == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
        ++num_correct_labels;
      }
    }
  }

  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / 100.0, 1e-4);
}

}  // namespace caffe
