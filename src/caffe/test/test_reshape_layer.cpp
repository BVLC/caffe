#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/reshape_layer.hpp"

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

TYPED_TEST(ReshapeLayerTest, TestFlattenOutputSizes) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(-1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3 * 6 * 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(ReshapeLayerTest, TestFlattenValues) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(-1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);
  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int c = 0; c < 3 * 6 * 5; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
        this->blob_bottom_->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    EXPECT_EQ(this->blob_top_->data_at(1, c, 0, 0),
        this->blob_bottom_->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
  }
}

// Test whether setting output dimensions to 0 either explicitly or implicitly
// copies the respective dimension of the input layer.
TYPED_TEST(ReshapeLayerTest, TestCopyDimensions) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

// When a dimension is set to -1, we should infer its value from the other
// dimensions (including those that get copied from below).
TYPED_TEST(ReshapeLayerTest, TestInferenceOfUnspecified) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(3);
  blob_shape->add_dim(10);
  blob_shape->add_dim(-1);

  // Count is 180, thus height should be 180 / (2*3*10) = 3.

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(ReshapeLayerTest, TestInferenceOfUnspecifiedWithStartAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reshape_param()->set_axis(1);
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(3);
  blob_shape->add_dim(10);
  blob_shape->add_dim(-1);

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 4);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(ReshapeLayerTest, TestInsertSingletonAxesStart) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reshape_param()->set_axis(0);
  layer_param.mutable_reshape_param()->set_num_axes(0);
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 7);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 1);
  EXPECT_EQ(this->blob_top_->shape(2), 1);
  EXPECT_EQ(this->blob_top_->shape(3), 2);
  EXPECT_EQ(this->blob_top_->shape(4), 3);
  EXPECT_EQ(this->blob_top_->shape(5), 6);
  EXPECT_EQ(this->blob_top_->shape(6), 5);
}

TYPED_TEST(ReshapeLayerTest, TestInsertSingletonAxesMiddle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reshape_param()->set_axis(2);
  layer_param.mutable_reshape_param()->set_num_axes(0);
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 7);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3);
  EXPECT_EQ(this->blob_top_->shape(2), 1);
  EXPECT_EQ(this->blob_top_->shape(3), 1);
  EXPECT_EQ(this->blob_top_->shape(4), 1);
  EXPECT_EQ(this->blob_top_->shape(5), 6);
  EXPECT_EQ(this->blob_top_->shape(6), 5);
}

TYPED_TEST(ReshapeLayerTest, TestInsertSingletonAxesEnd) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reshape_param()->set_axis(-1);
  layer_param.mutable_reshape_param()->set_num_axes(0);
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 7);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3);
  EXPECT_EQ(this->blob_top_->shape(2), 6);
  EXPECT_EQ(this->blob_top_->shape(3), 5);
  EXPECT_EQ(this->blob_top_->shape(4), 1);
  EXPECT_EQ(this->blob_top_->shape(5), 1);
  EXPECT_EQ(this->blob_top_->shape(6), 1);
}

TYPED_TEST(ReshapeLayerTest, TestFlattenMiddle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reshape_param()->set_axis(1);
  layer_param.mutable_reshape_param()->set_num_axes(2);
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(-1);

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3 * 6);
  EXPECT_EQ(this->blob_top_->shape(2), 5);
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
