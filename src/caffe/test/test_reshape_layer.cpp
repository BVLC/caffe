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
class ReshapeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  ReshapeLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ReshapeLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReshapeLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReshapeLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shared_ptr<ReshapeLayer<Dtype> > layer;

  shape->Clear();
  shape->add_dim(2 * 3 * 6 * 5);
  layer.reset(new ReshapeLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 1);
  EXPECT_EQ(this->blob_top_->shape(0), 2 * 3 * 6 * 5);

  shape->Clear();
  shape->add_dim(2 * 3 * 6);
  shape->add_dim(5);
  layer.reset(new ReshapeLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 2 * 3 * 6);
  EXPECT_EQ(this->blob_top_->shape(1), 5);

  shape->Clear();
  shape->add_dim(6);
  shape->add_dim(1);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(1);
  shape->add_dim(5);
  layer.reset(new ReshapeLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 6);
  EXPECT_EQ(this->blob_top_->shape(0), 6);
  EXPECT_EQ(this->blob_top_->shape(1), 1);
  EXPECT_EQ(this->blob_top_->shape(2), 2);
  EXPECT_EQ(this->blob_top_->shape(3), 3);
  EXPECT_EQ(this->blob_top_->shape(4), 1);
  EXPECT_EQ(this->blob_top_->shape(5), 5);
}

TYPED_TEST(ReshapeLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
}

TYPED_TEST(ReshapeLayerTest, TestForwardAfterReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // We know the above produced the correct result from TestForward.
  // Reshape the bottom and call layer.Reshape, then try again.
  vector<int> new_bottom_shape(1, 2 * 3 * 6 * 5);
  this->blob_bottom_->Reshape(new_bottom_shape);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
}

TYPED_TEST(ReshapeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  ReshapeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


}  // namespace caffe
