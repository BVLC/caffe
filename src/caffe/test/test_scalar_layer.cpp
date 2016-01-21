#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/scalar_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ScalarLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ScalarLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_eltwise_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_broadcast_0_(new Blob<Dtype>()),
        blob_bottom_broadcast_1_(new Blob<Dtype>()),
        blob_bottom_broadcast_2_(new Blob<Dtype>()),
        blob_bottom_scalar_(new Blob<Dtype>(vector<int>())),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    vector<int> broadcast_shape(2);
    broadcast_shape[0] = 2; broadcast_shape[1] = 3;
    this->blob_bottom_broadcast_0_->Reshape(broadcast_shape);
    broadcast_shape[0] = 3; broadcast_shape[1] = 4;
    this->blob_bottom_broadcast_1_->Reshape(broadcast_shape);
    broadcast_shape[0] = 4; broadcast_shape[1] = 5;
    this->blob_bottom_broadcast_2_->Reshape(broadcast_shape);
    FillerParameter filler_param;
    filler_param.set_min(1);
    filler_param.set_max(10);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_eltwise_);
    filler.Fill(this->blob_bottom_broadcast_0_);
    filler.Fill(this->blob_bottom_broadcast_1_);
    filler.Fill(this->blob_bottom_broadcast_2_);
    filler.Fill(this->blob_bottom_scalar_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ScalarLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_eltwise_;
    delete blob_bottom_broadcast_0_;
    delete blob_bottom_broadcast_1_;
    delete blob_bottom_broadcast_2_;
    delete blob_bottom_scalar_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_eltwise_;
  Blob<Dtype>* const blob_bottom_broadcast_0_;
  Blob<Dtype>* const blob_bottom_broadcast_1_;
  Blob<Dtype>* const blob_bottom_broadcast_2_;
  Blob<Dtype>* const blob_bottom_scalar_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ScalarLayerTest, TestDtypesAndDevices);

TYPED_TEST(ScalarLayerTest, TestForwardEltwise) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_eltwise_);
  LayerParameter layer_param;
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_eltwise_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[i], 1e-5);
  }
}

TYPED_TEST(ScalarLayerTest, TestForwardEltwiseInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_top_vec_[0] = this->blob_bottom_;  // in-place computation
  Blob<Dtype> orig_bottom(this->blob_bottom_->shape());
  orig_bottom.CopyFrom(*this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_eltwise_);
  LayerParameter layer_param;
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_bottom_->cpu_data();
  const int count = this->blob_bottom_->count();
  const Dtype* in_data_a = orig_bottom.cpu_data();
  const Dtype* in_data_b = this->blob_bottom_eltwise_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[i], 1e-5);
  }
}

TYPED_TEST(ScalarLayerTest, TestBackwardEltwiseInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  Blob<Dtype> orig_bottom(this->blob_bottom_->shape());
  orig_bottom.CopyFrom(*this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_eltwise_);
  LayerParameter layer_param;
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  Blob<Dtype> top_diff(this->blob_bottom_->shape());
  FillerParameter filler_param;
  filler_param.set_type("gaussian");
  filler_param.set_std(1);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(&top_diff);
  vector<bool> propagate_down(2, true);
  // Run forward + backward without in-place computation;
  // save resulting bottom diffs.
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_copy(top_diff.count(), top_diff.cpu_data(),
             this->blob_top_->mutable_cpu_diff());
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  const bool kReshape = true;
  const bool kCopyDiff = true;
  Blob<Dtype> orig_bottom_diff;
  orig_bottom_diff.CopyFrom(*this->blob_bottom_, kCopyDiff, kReshape);
  Blob<Dtype> orig_scalar_diff;
  orig_scalar_diff.CopyFrom(*this->blob_bottom_eltwise_,
                            kCopyDiff, kReshape);
  // Rerun forward + backward with in-place computation;
  // check that resulting bottom diffs are the same.
  this->blob_top_vec_[0] = this->blob_bottom_;  // in-place computation
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_copy(top_diff.count(), top_diff.cpu_data(),
             this->blob_bottom_->mutable_cpu_diff());
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(orig_bottom_diff.cpu_diff()[i],
                this->blob_bottom_->cpu_diff()[i], 1e-5);
  }
  for (int i = 0; i < this->blob_bottom_eltwise_->count(); ++i) {
    EXPECT_NEAR(orig_scalar_diff.cpu_diff()[i],
                this->blob_bottom_eltwise_->cpu_diff()[i], 1e-5);
  }
}

TYPED_TEST(ScalarLayerTest, TestForwardEltwiseWithParam) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScalarParameter* scalar_param = layer_param.mutable_scalar_param();
  scalar_param->set_axis(0);
  scalar_param->set_num_axes(-1);
  scalar_param->mutable_filler()->set_type("gaussian");
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_->cpu_data();
  const Dtype* in_data_b = layer->blobs()[0]->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[i], 1e-5);
  }
}

TYPED_TEST(ScalarLayerTest, TestForwardBroadcastBegin) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_0_);
  LayerParameter layer_param;
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_top_->data_at(n, c, h, w),
                      this->blob_bottom_->data_at(n, c, h, w) *
                      this->blob_bottom_broadcast_0_->data_at(n, c, 0, 0),
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScalarLayerTest, TestForwardBroadcastMiddle) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_1_);
  LayerParameter layer_param;
  layer_param.mutable_scalar_param()->set_axis(1);
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_top_->data_at(n, c, h, w),
                      this->blob_bottom_->data_at(n, c, h, w) *
                      this->blob_bottom_broadcast_1_->data_at(c, h, 0, 0),
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScalarLayerTest, TestForwardBroadcastMiddleInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_top_vec_[0] = this->blob_bottom_;  // in-place computation
  Blob<Dtype> orig_bottom(this->blob_bottom_->shape());
  orig_bottom.CopyFrom(*this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_1_);
  LayerParameter layer_param;
  layer_param.mutable_scalar_param()->set_axis(1);
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_bottom_->data_at(n, c, h, w),
                      orig_bottom.data_at(n, c, h, w) *
                      this->blob_bottom_broadcast_1_->data_at(c, h, 0, 0),
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScalarLayerTest, TestBackwardBroadcastMiddleInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  Blob<Dtype> orig_bottom(this->blob_bottom_->shape());
  orig_bottom.CopyFrom(*this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_1_);
  LayerParameter layer_param;
  layer_param.mutable_scalar_param()->set_axis(1);
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  Blob<Dtype> top_diff(this->blob_bottom_->shape());
  FillerParameter filler_param;
  filler_param.set_type("gaussian");
  filler_param.set_std(1);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(&top_diff);
  vector<bool> propagate_down(2, true);
  // Run forward + backward without in-place computation;
  // save resulting bottom diffs.
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_copy(top_diff.count(), top_diff.cpu_data(),
             this->blob_top_->mutable_cpu_diff());
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  const bool kReshape = true;
  const bool kCopyDiff = true;
  Blob<Dtype> orig_bottom_diff;
  orig_bottom_diff.CopyFrom(*this->blob_bottom_, kCopyDiff, kReshape);
  Blob<Dtype> orig_scalar_diff;
  orig_scalar_diff.CopyFrom(*this->blob_bottom_broadcast_1_,
                            kCopyDiff, kReshape);
  // Rerun forward + backward with in-place computation;
  // check that resulting bottom diffs are the same.
  this->blob_top_vec_[0] = this->blob_bottom_;  // in-place computation
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_copy(top_diff.count(), top_diff.cpu_data(),
             this->blob_bottom_->mutable_cpu_diff());
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(orig_bottom_diff.cpu_diff()[i],
                this->blob_bottom_->cpu_diff()[i], 1e-5);
  }
  for (int i = 0; i < this->blob_bottom_broadcast_1_->count(); ++i) {
    EXPECT_NEAR(orig_scalar_diff.cpu_diff()[i],
                this->blob_bottom_broadcast_1_->cpu_diff()[i], 1e-5);
  }
}

TYPED_TEST(ScalarLayerTest, TestForwardBroadcastMiddleWithParam) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScalarParameter* scalar_param = layer_param.mutable_scalar_param();
  scalar_param->set_axis(1);
  scalar_param->set_num_axes(2);
  scalar_param->mutable_filler()->set_type("gaussian");
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_top_->data_at(n, c, h, w),
                      this->blob_bottom_->data_at(n, c, h, w) *
                      layer->blobs()[0]->data_at(c, h, 0, 0), 1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScalarLayerTest, TestForwardBroadcastMiddleWithParamAndBias) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScalarParameter* scalar_param = layer_param.mutable_scalar_param();
  scalar_param->set_axis(1);
  scalar_param->set_num_axes(2);
  scalar_param->mutable_filler()->set_type("gaussian");
  scalar_param->set_bias_term(true);
  scalar_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_top_->data_at(n, c, h, w),
                      this->blob_bottom_->data_at(n, c, h, w) *
                      layer->blobs()[0]->data_at(c, h, 0, 0) +
                      layer->blobs()[1]->data_at(c, h, 0, 0), 1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScalarLayerTest, TestForwardBroadcastEnd) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_2_);
  LayerParameter layer_param;
  layer_param.mutable_scalar_param()->set_axis(2);
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_top_->data_at(n, c, h, w),
                      this->blob_bottom_->data_at(n, c, h, w) *
                      this->blob_bottom_broadcast_2_->data_at(h, w, 0, 0),
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScalarLayerTest, TestForwardScalar) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scalar_);
  LayerParameter layer_param;
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data = this->blob_bottom_->cpu_data();
  const Dtype scalar = *this->blob_bottom_scalar_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data[i] * scalar, 1e-5);
  }
}

TYPED_TEST(ScalarLayerTest, TestForwardScalarAxis2) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scalar_);
  LayerParameter layer_param;
  layer_param.mutable_scalar_param()->set_axis(2);
  shared_ptr<ScalarLayer<Dtype> > layer(new ScalarLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data = this->blob_bottom_->cpu_data();
  const Dtype scalar = *this->blob_bottom_scalar_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data[i] * scalar, 1e-5);
  }
}

TYPED_TEST(ScalarLayerTest, TestGradientEltwise) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_eltwise_);
  LayerParameter layer_param;
  ScalarLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ScalarLayerTest, TestGradientEltwiseWithParam) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScalarParameter* scalar_param = layer_param.mutable_scalar_param();
  scalar_param->set_axis(0);
  scalar_param->set_num_axes(-1);
  scalar_param->mutable_filler()->set_type("gaussian");
  ScalarLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ScalarLayerTest, TestGradientBroadcastBegin) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_0_);
  LayerParameter layer_param;
  ScalarLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ScalarLayerTest, TestGradientBroadcastMiddle) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_1_);
  LayerParameter layer_param;
  layer_param.mutable_scalar_param()->set_axis(1);
  ScalarLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ScalarLayerTest, TestGradientBroadcastMiddleWithParam) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_1_);
  LayerParameter layer_param;
  ScalarParameter* scalar_param = layer_param.mutable_scalar_param();
  scalar_param->set_axis(1);
  scalar_param->set_num_axes(2);
  scalar_param->mutable_filler()->set_type("gaussian");
  ScalarLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ScalarLayerTest, TestGradientBroadcastEnd) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_2_);
  LayerParameter layer_param;
  layer_param.mutable_scalar_param()->set_axis(2);
  ScalarLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ScalarLayerTest, TestGradientScalar) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scalar_);
  LayerParameter layer_param;
  ScalarLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ScalarLayerTest, TestGradientScalarAndBias) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scalar_);
  LayerParameter layer_param;
  ScalarParameter* scalar_param = layer_param.mutable_scalar_param();
  scalar_param->set_bias_term(true);
  scalar_param->mutable_bias_filler()->set_type("gaussian");
  ScalarLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ScalarLayerTest, TestGradientScalarAxis2) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scalar_);
  LayerParameter layer_param;
  layer_param.mutable_scalar_param()->set_axis(2);
  ScalarLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
