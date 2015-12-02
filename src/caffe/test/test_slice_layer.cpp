#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/slice_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SliceLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SliceLayerTest()
      : blob_bottom_(new Blob<Dtype>(6, 12, 2, 3)),
        blob_top_0_(new Blob<Dtype>()),
        blob_top_1_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_0_.push_back(blob_top_0_);
    blob_top_vec_0_.push_back(blob_top_1_);
    blob_top_vec_1_.push_back(blob_top_0_);
    blob_top_vec_1_.push_back(blob_top_1_);
    blob_top_vec_1_.push_back(blob_top_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
  }

  virtual void ReduceBottomBlobSize() {
    blob_bottom_->Reshape(4, 5, 2, 2);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
  }

  virtual ~SliceLayerTest() {
    delete blob_top_0_; delete blob_top_1_;
    delete blob_top_2_; delete blob_bottom_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_0_;
  Blob<Dtype>* const blob_top_1_;
  Blob<Dtype>* const blob_top_2_;
  vector<Blob<Dtype>*> blob_top_vec_0_, blob_top_vec_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

TYPED_TEST_CASE(SliceLayerTest, TestDtypesAndDevices);

TYPED_TEST(SliceLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_axis(0);
  SliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_1_);
  EXPECT_EQ(this->blob_bottom_->num(), 3 * this->blob_top_0_->num());
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_top_1_->num());
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_top_2_->num());
  EXPECT_EQ(this->blob_bottom_->channels(), this->blob_top_0_->channels());
  EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_0_->height());
  EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_0_->width());
}

TYPED_TEST(SliceLayerTest, TestSetupChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->add_slice_point(3);
  SliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_0_);
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_0_->channels(), 3);
  EXPECT_EQ(this->blob_top_1_->channels(), 9);
  EXPECT_EQ(this->blob_bottom_->channels(),
    this->blob_top_0_->channels() + this->blob_top_1_->channels());
  EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_0_->height());
  EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_0_->width());
}

TYPED_TEST(SliceLayerTest, TestTrivialSlice) {
  // Test the trivial (single output) "slice" operation --
  // should be the identity.
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SliceLayer<Dtype> layer(layer_param);
  this->blob_top_vec_0_.resize(1);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_0_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_0_->shape());
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_bottom_->cpu_data()[i],
              this->blob_top_0_->cpu_data()[i]);
  }
}

TYPED_TEST(SliceLayerTest, TestSliceAcrossNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_axis(0);
  SliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_0_);
  const int top_num = this->blob_bottom_->num() / 2;
  ASSERT_EQ(top_num, this->blob_top_0_->num());
  ASSERT_EQ(top_num, this->blob_top_1_->num());
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_0_);
  for (int n = 0; n < top_num; ++n) {
    for (int c = 0; c < this->blob_top_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
                    this->blob_top_0_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n + 3, c, h, w),
                    this->blob_top_1_->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(SliceLayerTest, TestSliceAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Slice at 2, 8: should produce output blobs with #channels 2, 6, 4.
  const int kSlicePoint0 = 2;
  const int kSlicePoint1 = 8;
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint0);
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint1);
  SliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_1_);
  ASSERT_EQ(kSlicePoint0, this->blob_top_0_->channels());
  ASSERT_EQ(kSlicePoint1 - kSlicePoint0, this->blob_top_1_->channels());
  ASSERT_EQ(this->blob_bottom_->channels() - kSlicePoint1,
            this->blob_top_2_->channels());
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_1_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_top_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
              this->blob_top_0_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c + kSlicePoint0, h, w),
              this->blob_top_1_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_2_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c + kSlicePoint1, h, w),
              this->blob_top_2_->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(SliceLayerTest, TestGradientTrivial) {
  // Test the trivial (single output) "slice" operation --
  // should be the identity.
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SliceLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  this->blob_top_vec_0_.resize(1);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_0_);
}

TYPED_TEST(SliceLayerTest, TestGradientAcrossNum) {
  typedef typename TypeParam::Dtype Dtype;
  // Gradient checks are slow; reduce blob size.
  this->ReduceBottomBlobSize();
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_axis(0);
  SliceLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_0_);
}

TYPED_TEST(SliceLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  // Gradient checks are slow; reduce blob size.
  this->ReduceBottomBlobSize();
  LayerParameter layer_param;
  const int kSlicePoint = 4;
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint);
  SliceLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_0_);
}

}  // namespace caffe
