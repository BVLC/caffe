#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ReverseLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ReverseLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(11, 3, 6, 5)),
        blob_bottom_0_seg_(new Blob<Dtype>(6, 1, 1, 1)),
        blob_bottom_1_(new Blob<Dtype>(10, 3, 6, 5)),
        blob_bottom_1_seg_(new Blob<Dtype>(5, 1, 1, 1)),
        blob_bottom_2_(new Blob<Dtype>(5, 3, 6, 5)),
        blob_bottom_2_seg_(new Blob<Dtype>(5, 1, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    Dtype* segment_data = this->blob_bottom_0_seg_->mutable_cpu_data();
    segment_data[0] = 0; segment_data[1] = 4; segment_data[2] = 6;
    segment_data[3] = 4; segment_data[4] = -1; segment_data[5] = -1;
    filler.Fill(this->blob_bottom_1_);
    segment_data = this->blob_bottom_1_seg_->mutable_cpu_data();
    segment_data[0] = 0; segment_data[1] = 3; segment_data[2] = 4;
    segment_data[3] = 3; segment_data[4] = 8;
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_0_.push_back(blob_bottom_0_seg_);
    blob_bottom_vec_1_.push_back(blob_bottom_1_);
    blob_bottom_vec_1_.push_back(blob_bottom_1_seg_);
    blob_bottom_vec_2_.push_back(blob_bottom_2_);
    blob_bottom_vec_2_.push_back(blob_bottom_2_seg_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ReverseLayerTest() {
    delete blob_bottom_0_; delete blob_bottom_0_seg_;
    delete blob_bottom_1_; delete blob_bottom_1_seg_;
    delete blob_bottom_2_; delete blob_bottom_2_seg_;
    delete blob_top_;
  }

  void CheckReverse(const Blob<Dtype>* const blob1,
      const Blob<Dtype>* const blob2,
      const int segment_start, const int segment_end) {
    CheckBlobValues(blob1, blob2, segment_start,
        segment_end, true);
  }

  void CheckIdentical(const Blob<Dtype>* const blob1,
      const Blob<Dtype>* const blob2,
      const int segment_start, const int segment_end) {
    CheckBlobValues(blob1, blob2, segment_start,
        segment_end, false);
  }

  void CheckBlobValues(const Blob<Dtype>* const blob1,
      const Blob<Dtype>* const blob2,
      const int segment_start, const int segment_end,
      const bool reverse) {
    int segment_len = segment_end - segment_start;
    for (int n = 0; n < segment_len; ++n) {
      for (int c = 0; c < blob1->channels(); ++c) {
        for (int h = 0; h < blob1->height(); ++h) {
          for (int w = 0; w < blob1->width(); ++w) {
            if (reverse) {
              EXPECT_EQ(blob1->data_at(segment_start+n, c, h, w),
                  blob2->data_at(segment_end-1-n, c, h, w));
            } else {
              EXPECT_EQ(blob1->data_at(segment_start+n, c, h, w),
                  blob2->data_at(segment_start+n, c, h, w));
            }
          }
        }
      }
    }
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_0_seg_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_1_seg_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_2_seg_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_0_;
  vector<Blob<Dtype>*> blob_bottom_vec_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_2_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReverseLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReverseLayerTest, TestForwardWithoutSegment) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReverseLayer<Dtype> layer(layer_param);

  // odd number input test
  this->blob_bottom_vec_0_.resize(1);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);
  this->CheckReverse(this->blob_bottom_vec_0_[0], this->blob_top_,
      0, this->blob_bottom_vec_0_[0]->shape(0));

  // even number input test
  this->blob_bottom_vec_1_.resize(1);
  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_1_, this->blob_top_vec_);
  this->CheckReverse(this->blob_bottom_vec_1_[0], this->blob_top_,
      0, this->blob_bottom_vec_1_[0]->shape(0));
}

TYPED_TEST(ReverseLayerTest, TestForwardWithSegment) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReverseLayer<Dtype> layer(layer_param);

  // odd number input test
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);
  int segment_start = 0;
  int segment_end = 4;
  this->CheckReverse(this->blob_bottom_vec_0_[0], this->blob_top_,
      segment_start, segment_end);
  segment_start = 4;
  segment_end = 6;
  this->CheckIdentical(this->blob_bottom_vec_0_[0], this->blob_top_,
      segment_start, segment_end);
  segment_start = 6;
  segment_end = 10;
  this->CheckReverse(this->blob_bottom_vec_0_[0], this->blob_top_,
      segment_start, segment_end);
  segment_start = 10;
  segment_end = 11;
  this->CheckIdentical(this->blob_bottom_vec_0_[0], this->blob_top_,
      segment_start, segment_end);

  // even number input test
  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_1_, this->blob_top_vec_);
  segment_start = 0;
  segment_end = 3;
  this->CheckReverse(this->blob_bottom_vec_1_[0], this->blob_top_,
      segment_start, segment_end);
  segment_start = 3;
  segment_end = 4;
  this->CheckIdentical(this->blob_bottom_vec_1_[0], this->blob_top_,
      segment_start, segment_end);
  segment_start = 4;
  segment_end = 7;
  this->CheckReverse(this->blob_bottom_vec_1_[0], this->blob_top_,
      segment_start, segment_end);
  segment_start = 7;
  segment_end = 10;
  this->CheckIdentical(this->blob_bottom_vec_1_[0], this->blob_top_,
      segment_start, segment_end);

  // zero reverse test
  layer.SetUp(this->blob_bottom_vec_2_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_2_, this->blob_top_vec_);
  segment_start = 0;
  segment_end = 5;
  this->CheckIdentical(this->blob_bottom_vec_2_[0], this->blob_top_,
      segment_start, segment_end);
}

TYPED_TEST(ReverseLayerTest, TestGradientWithoutSegmentOdd) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReverseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  this->blob_bottom_vec_0_.resize(1);
  checker.CheckGradient(&layer, this->blob_bottom_vec_0_,
      this->blob_top_vec_);
}

TYPED_TEST(ReverseLayerTest, TestGradientWithoutSegmentEven) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReverseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  this->blob_bottom_vec_1_.resize(1);
  checker.CheckGradient(&layer, this->blob_bottom_vec_1_,
      this->blob_top_vec_);
}

TYPED_TEST(ReverseLayerTest, TestGradientWithSegmentOdd) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReverseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_0_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(ReverseLayerTest, TestGradientWithSegmentEven) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReverseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_1_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(ReverseLayerTest, TestGradientZereReverse) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReverseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_2_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
