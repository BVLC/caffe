#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class SumLayerTest : public ::testing::Test {
 protected:
  SumLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(100, 2, 3, 4)),
        blob_top_(new Blob<Dtype>()) {
    // fill the probability values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SumLayerTest() {
    delete blob_bottom_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SumLayerTest, TestDtypes);

TYPED_TEST(SumLayerTest, TestSetup) {
  LayerParameter layer_param;
  SumLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 100);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(SumLayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  SumLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  for (int i = 0; i < 100; ++i) {
    double sum = 0;
    for (int c = 0; c < 2; ++c) {
      for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 4; ++w) {
          sum += this->blob_bottom_data_->data_at(i, c, h, w);
        }
      }
    }
    EXPECT_NEAR(this->blob_top_->data_at(i, 0, 0, 0), sum, 1e-4);
  }
}

}  // namespace caffe
