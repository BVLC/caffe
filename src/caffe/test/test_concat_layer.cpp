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
class ConcatLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConcatLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_bottom_1_(new Blob<Dtype>(2, 5, 6, 5)),
        blob_bottom_2_(new Blob<Dtype>(5, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    shared_ptr<ConstantFiller<Dtype> > filler;
    FillerParameter filler_param;
    filler_param.set_value(1.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_0_);
    filler_param.set_value(2.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_1_);
    filler_param.set_value(3.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_2_);
    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_0_.push_back(blob_bottom_1_);
    blob_bottom_vec_1_.push_back(blob_bottom_0_);
    blob_bottom_vec_1_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ConcatLayerTest() {
    delete blob_bottom_0_; delete blob_bottom_1_;
    delete blob_bottom_2_; delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_0_, blob_bottom_vec_1_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConcatLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConcatLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_concat_param()->set_axis(0);
  ConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(),
      this->blob_bottom_0_->num() + this->blob_bottom_2_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_0_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0_->width());
}

TYPED_TEST(ConcatLayerTest, TestSetupChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0_->num());
  EXPECT_EQ(this->blob_top_->channels(),
      this->blob_bottom_0_->channels() + this->blob_bottom_1_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0_->width());
}

TYPED_TEST(ConcatLayerTest, TestSetupChannelsNegativeIndexing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  // "channels" index is the third one from the end -- test negative indexing
  // by setting axis to -3 and checking that we get the same results as above in
  // TestSetupChannels.
  layer_param.mutable_concat_param()->set_axis(-3);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0_->num());
  EXPECT_EQ(this->blob_top_->channels(),
      this->blob_bottom_0_->channels() + this->blob_bottom_1_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0_->width());
}

TYPED_TEST(ConcatLayerTest, TestForwardTrivial) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  this->blob_bottom_vec_0_.resize(1);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_0_->count(); ++i) {
    EXPECT_EQ(this->blob_bottom_0_->cpu_data()[i],
              this->blob_top_->cpu_data()[i]);
  }
}

TYPED_TEST(ConcatLayerTest, TestForwardNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_concat_param()->set_axis(0);
  ConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_1_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_vec_1_[0]->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
              this->blob_bottom_vec_1_[0]->data_at(n, c, h, w));
        }
      }
    }
  }
  for (int n = 0; n < this->blob_bottom_vec_1_[1]->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n + 2, c, h, w),
              this->blob_bottom_vec_1_[1]->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(ConcatLayerTest, TestForwardChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
              this->blob_bottom_vec_0_[0]->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_bottom_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n, c + 3, h, w),
              this->blob_bottom_vec_0_[1]->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(ConcatLayerTest, TestGradientTrivial) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  this->blob_bottom_vec_0_.resize(1);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_0_,
      this->blob_top_vec_);
}

TYPED_TEST(ConcatLayerTest, TestGradientNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_concat_param()->set_axis(0);
  ConcatLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_1_,
    this->blob_top_vec_);
}

TYPED_TEST(ConcatLayerTest, TestGradientChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_0_,
    this->blob_top_vec_);
}

TYPED_TEST(ConcatLayerTest, TestGradientChannelsBottomOneOnly) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_0_,
    this->blob_top_vec_, 1);
}

}  // namespace caffe
