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
      : blob_bottom_0_(new Blob<Dtype>(2, 5, 6, 5)),
        blob_bottom_1_(new Blob<Dtype>(2, 4, 5, 3)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    for (int i = 0; i < this->blob_bottom_0_->count(); ++i) {
      this->blob_bottom_0_->mutable_cpu_data()[i] = i;
    }


    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_0_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~CropLayerTest() {
    delete blob_bottom_0_; delete blob_bottom_1_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_0_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


TYPED_TEST_CASE(CropLayerTest, TestDtypesAndDevices);

TYPED_TEST(CropLayerTest, TestSetupShapeAll) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  // Crop all dimensions
  layer_param.mutable_crop_param()->set_axis(0);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
  }
}

TYPED_TEST(CropLayerTest, TestSetupShapeDefault) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop last two dimensions, axis is 2 by default
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
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
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 3) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}


TYPED_TEST(CropLayerTest, TestForwardNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);

  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);
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

TYPED_TEST(CropLayerTest, TestForwardNumOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);
  layer_param.mutable_crop_param()->add_offset(0);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if ( n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3) ) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n+0, c+1, h+1, w+2));
          }
        }
      }
    }
  }
}

TYPED_TEST(CropLayerTest, TestGradientNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CropLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);

  // Copy top data into diff
  caffe_copy(this->blob_top_->count(), this->blob_top_->cpu_data(),
             this->blob_top_->mutable_cpu_diff());

  // Do backward pass
  vector<bool> propagate_down(2, true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_0_);


  // Check results
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if ( n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3) ) {
            EXPECT_EQ(this->blob_bottom_0_->diff_at(n, c, h, w),
                      this->blob_bottom_0_->data_at(n, c, h, w));
          } else {
            EXPECT_EQ(this->blob_bottom_0_->diff_at(n, c, h, w), 0);
          }
        }
      }
    }
  }
}

TYPED_TEST(CropLayerTest, TestGradientNumOffset) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);
  layer_param.mutable_crop_param()->add_offset(0);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  CropLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);

  // Copy top data into diff
  caffe_copy(this->blob_top_->count(), this->blob_top_->cpu_data(),
             this->blob_top_->mutable_cpu_diff());

  // Do backward pass
  vector<bool> propagate_down(2, true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_0_);


  // Check results
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if ( 0 <= n && n < 0 + this->blob_top_->shape(0) &&
              1 <= c && c < 1 + this->blob_top_->shape(1) &&
              1 <= h && h < 1 + this->blob_top_->shape(2) &&
              2 <= w && w < 2 + this->blob_top_->shape(3) ) {
            EXPECT_EQ(this->blob_bottom_0_->diff_at(n, c, h, w),
                      this->blob_bottom_0_->data_at(n, c, h, w));
          } else {
            EXPECT_EQ(this->blob_bottom_0_->diff_at(n, c, h, w), 0);
          }
        }
      }
    }
  }
}

}  // namespace caffe
