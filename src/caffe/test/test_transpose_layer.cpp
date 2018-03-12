#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/transpose_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

  template <typename TypeParam>
  class TransposeLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
   protected:
    TransposeLayerTest()
        : blob_bottom_(new Blob<Dtype>(5, 2, 3, 4)),
          blob_top_(new Blob<Dtype>()) {
      // fill the values
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~TransposeLayerTest() { delete blob_bottom_; delete blob_top_; }
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
  };

  TYPED_TEST_CASE(TransposeLayerTest, TestDtypesAndDevices);

  TYPED_TEST(TransposeLayerTest, TestTopShape) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_transpose_param()->add_dim(0);
    layer_param.mutable_transpose_param()->add_dim(2);
    layer_param.mutable_transpose_param()->add_dim(3);
    layer_param.mutable_transpose_param()->add_dim(1);
    TransposeLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    vector<int> bottom_shape = this->blob_bottom_->shape();
    vector<int> top_shape = this->blob_top_->shape();
    EXPECT_EQ(top_shape.at(0), bottom_shape.at(0));
    EXPECT_EQ(top_shape.at(1), bottom_shape.at(2));
    EXPECT_EQ(top_shape.at(2), bottom_shape.at(3));
    EXPECT_EQ(top_shape.at(3), bottom_shape.at(1));
  }

  TYPED_TEST(TransposeLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_transpose_param()->add_dim(0);
    layer_param.mutable_transpose_param()->add_dim(2);
    layer_param.mutable_transpose_param()->add_dim(3);
    layer_param.mutable_transpose_param()->add_dim(1);
    TransposeLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    for (int n = 0; n < this->blob_top_->shape(0); n++) {
      for (int c = 0; c < this->blob_top_->shape(1); c++) {
        for (int h = 0; h < this->blob_top_->shape(2); h++) {
          for (int w = 0; w < this->blob_top_->shape(3); w++) {
            EXPECT_EQ(*(bottom_data + this->blob_bottom_->offset(n, w, c, h)),
              *top_data);
            top_data += 1;
          }
        }
      }
    }
  }

  TYPED_TEST(TransposeLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_transpose_param()->add_dim(0);
    layer_param.mutable_transpose_param()->add_dim(2);
    layer_param.mutable_transpose_param()->add_dim(3);
    layer_param.mutable_transpose_param()->add_dim(1);
    TransposeLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-4);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

}  // namespace caffe
