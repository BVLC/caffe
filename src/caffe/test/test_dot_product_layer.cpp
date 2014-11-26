#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class DotProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DotProductLayerTest()
      : blob_bottom_a_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_b_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DotProductLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_b_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DotProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(DotProductLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DotProductLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(DotProductLayerTest, TestDotProduct) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DotProductLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int num = this->blob_bottom_a_->num();
  const int dim = this->blob_bottom_a_->count() / num;
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  for (int i = 0; i < num; ++i) {
    Dtype sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += in_data_a[i*dim + j] * in_data_b[i*dim + j];
    }
    EXPECT_FLOAT_EQ(data[i], sum);
  }
}

TYPED_TEST(DotProductLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DotProductLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
