#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/crop_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CropLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CropLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 4, 5, 4)),
        blob_bottom_1_(new Blob<Dtype>(2, 3, 4, 2)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);

    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~CropLayerTest() {
    delete blob_bottom_0_; delete blob_bottom_1_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


TYPED_TEST_CASE(CropLayerTest, TestDtypesAndDevices);

TYPED_TEST(CropLayerTest, TestSetupShapeAll) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop all dimensions
  layer_param.mutable_crop_param()->set_axis(0);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
  }
}

TYPED_TEST(CropLayerTest, TestSetupShapeDefault) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop last two dimensions, axis is 2 by default
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 2) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(CropLayerTest, TestSetupShapeNegativeIndexing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop last dimension by negative indexing
  layer_param.mutable_crop_param()->set_axis(-1);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 3) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(CropLayerTest, TestDimensionsCheck) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Reshape size blob to have incompatible sizes for uncropped dimensions:
  // the size blob has more channels than the data blob, but this is fine
  // since the channels dimension is not cropped in this configuration.
  this->blob_bottom_1_->Reshape(2, 5, 4, 2);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 2) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(CropLayerTest, TestCropAll) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if ( n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3) ) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n, c, h, w));
          }
        }
      }
    }
  }
}

TYPED_TEST(CropLayerTest, TestCropAllOffset) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);
  layer_param.mutable_crop_param()->add_offset(0);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if ( n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3) ) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n, c+1, h+1, w+2));
          }
        }
      }
    }
  }
}

TYPED_TEST(CropLayerTest, TestCropHW) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(2);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if (n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3)) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n, c, h+1, w+2));
          }
        }
      }
    }
  }
}

TYPED_TEST(CropLayerTest, TestCrop5D) {
  typedef typename TypeParam::Dtype Dtype;
  // Add dimension to each bottom for >4D check
  vector<int> bottom_0_shape = this->blob_bottom_0_->shape();
  vector<int> bottom_1_shape = this->blob_bottom_1_->shape();
  bottom_0_shape.push_back(2);
  bottom_1_shape.push_back(1);
  this->blob_bottom_0_->Reshape(bottom_0_shape);
  this->blob_bottom_1_->Reshape(bottom_1_shape);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_0_);
  filler.Fill(this->blob_bottom_1_);
  // Make layer
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(2);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  layer_param.mutable_crop_param()->add_offset(0);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<int> bottom_idx = vector<int>(5, 0);
  vector<int> top_idx = vector<int>(5, 0);
  for (int n = 0; n < this->blob_bottom_0_->shape(0); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->shape(1); ++c) {
      for (int z = 0; z < this->blob_bottom_0_->shape(2); ++z) {
        for (int h = 0; h < this->blob_bottom_0_->shape(3); ++h) {
          for (int w = 0; w < this->blob_bottom_0_->shape(4); ++w) {
            if (n < this->blob_top_->shape(0) &&
                c < this->blob_top_->shape(1) &&
                z < this->blob_top_->shape(2) &&
                h < this->blob_top_->shape(3) &&
                w < this->blob_top_->shape(4)) {
              bottom_idx[0] = top_idx[0] = n;
              bottom_idx[1] = top_idx[1] = c;
              bottom_idx[2] = z;
              bottom_idx[3] = h;
              bottom_idx[4] = top_idx[4] = w;
              top_idx[2] = z+1;
              top_idx[3] = h+2;
              EXPECT_EQ(this->blob_top_->data_at(bottom_idx),
                  this->blob_bottom_0_->data_at(top_idx));
            }
          }
        }
      }
    }
  }
}

TYPED_TEST(CropLayerTest, TestCropAllGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);
  CropLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CropLayerTest, TestCropHWGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(2);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  CropLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CropLayerTest, TestCrop5DGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(2);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  layer_param.mutable_crop_param()->add_offset(0);
  CropLayer<Dtype> layer(layer_param);
  // Add dimension to each bottom for >4D check
  vector<int> bottom_0_shape = this->blob_bottom_0_->shape();
  vector<int> bottom_1_shape = this->blob_bottom_1_->shape();
  bottom_0_shape.push_back(2);
  bottom_1_shape.push_back(1);
  this->blob_bottom_0_->Reshape(bottom_0_shape);
  this->blob_bottom_1_->Reshape(bottom_1_shape);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
