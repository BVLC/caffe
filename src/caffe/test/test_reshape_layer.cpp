#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"

#include "caffe/proto/reshape_param.pb.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ReshapeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ReshapeLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    layer_param_.MutableExtension(reshape_param)->mutable_shape()->
        add_dim(2 * 3 * 4 * 5);
  }
  virtual ~ReshapeLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  LayerParameter layer_param_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReshapeLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReshapeLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  shared_ptr<ReshapeLayer<Dtype> > layer(
      new ReshapeLayer<Dtype>(this->layer_param_));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(1, this->blob_top_->num_axes());
  EXPECT_EQ(2 * 3 * 4 * 5, this->blob_top_->shape(0));
}

TYPED_TEST(ReshapeLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  shared_ptr<ReshapeLayer<Dtype> > layer(
      new ReshapeLayer<Dtype>(this->layer_param_));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const int count = this->blob_bottom_->count();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

TYPED_TEST(ReshapeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  ReshapeLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
