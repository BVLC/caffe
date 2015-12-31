#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/math_functions.hpp"

#ifndef CPU_ONLY  // CPU-GPU test

namespace caffe {

template<typename TypeParam>
class PoolingNDSKLayerTest : public GPUDeviceTest<TypeParam> {
 protected:
  PoolingNDSKLayerTest()
      : blob_bottom_(new Blob<TypeParam>()),
        blob_top_(new Blob<TypeParam>()) {
  }

  virtual void SetUp() {
    BlobShape shape;
    shape.add_dim(1);  // Batch
    shape.add_dim(1);  // Channels
    shape.add_dim(5);  // Depth
    shape.add_dim(5);  // Height
    shape.add_dim(5);  // Width
    blob_bottom_->Reshape(shape);

    shape.add_dim(1);  // Batch
    shape.add_dim(1);  // Channels
    shape.add_dim(1);  // Depth
    shape.add_dim(1);  // Height
    shape.add_dim(1);  // Width
    blob_top_->Reshape(shape);

    // fill the values
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~PoolingNDSKLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  void TestForward() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param =
        layer_param.mutable_pooling_param();

    pooling_param->add_kernel_size(3);
    pooling_param->add_kernel_size(3);
    pooling_param->add_kernel_size(3);

    pooling_param->add_dilation(2);
    pooling_param->add_dilation(2);
    pooling_param->add_dilation(2);

    pooling_param->set_axis(1);

    PoolingLayer<TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    int_tp d = blob_bottom_->shape(2);
    int_tp h = blob_bottom_->shape(3);
    int_tp w = blob_bottom_->shape(4);

    TypeParam *bottom_data = blob_bottom_->mutable_cpu_data();

    TypeParam maxval = 0;

    for (int_tp cd = 0; cd < d; ++cd) {
      for (int_tp ch = 0; ch < h; ++ch) {
        for (int_tp cw = 0; cw < w; ++cw) {
          bottom_data[cw + ch * w + cd * w * h] =
              cw + ch * w + cd * w * h;
          if (cw % 2 == 0 && ch % 2 == 0 && cd % 2 == 0) {
            maxval = std::max((TypeParam)(cw + ch * w + cd * w * h), maxval);
          }
        }
      }
    }

    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const TypeParam *top_data = blob_top_->cpu_data();

    EXPECT_EQ(maxval, top_data[0]);
  }

  void TestBackward() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param =
        layer_param.mutable_pooling_param();

    pooling_param->add_kernel_size(3);
    pooling_param->add_kernel_size(3);
    pooling_param->add_kernel_size(3);

    pooling_param->add_dilation(2);
    pooling_param->add_dilation(2);
    pooling_param->add_dilation(2);

    pooling_param->set_axis(1);

    PoolingLayer<TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    int_tp d = blob_bottom_->shape(2);
    int_tp h = blob_bottom_->shape(3);
    int_tp w = blob_bottom_->shape(4);

    TypeParam *bottom_data = blob_bottom_->mutable_cpu_data();

    TypeParam maxval = 0;

    for (int_tp cd = 0; cd < d; ++cd) {
      for (int_tp ch = 0; ch < h; ++ch) {
        for (int_tp cw = 0; cw < w; ++cw) {
          bottom_data[cw + ch * w + cd * w * h] =
              cw + ch * w + cd * w * h;
          if (cw % 2 == 0 && ch % 2 == 0 && cd % 2 == 0) {
            maxval = std::max((TypeParam)(cw + ch * w + cd * w * h), maxval);
          }
        }
      }
    }

    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    TypeParam *top_diff = blob_top_->mutable_cpu_diff();
    top_diff[0] = maxval;

    std::vector<bool> prop_down;
    prop_down.push_back(true);

    layer.Backward(this->blob_top_vec_, prop_down, this->blob_bottom_vec_);

    const TypeParam *bottom_diff = blob_bottom_->cpu_diff();

    for (int_tp cd = 0; cd < d; ++cd) {
      for (int_tp ch = 0; ch < h; ++ch) {
        for (int_tp cw = 0; cw < w; ++cw) {
          if (maxval == cw + ch * w + cd * w * h) {
            EXPECT_EQ(maxval, bottom_diff[cw + ch * w + cd * w * h]);
          }
        }
      }
    }
  }

  Blob<TypeParam>* const blob_bottom_;
  Blob<TypeParam>* const blob_top_;

  vector<Blob<TypeParam>*> blob_bottom_vec_;
  vector<Blob<TypeParam>*> blob_top_vec_;
};

TYPED_TEST_CASE(PoolingNDSKLayerTest, TestDtypes);

TYPED_TEST(PoolingNDSKLayerTest, TestSetup) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param =
      layer_param.mutable_pooling_param();

  pooling_param->add_kernel_size(3);
  pooling_param->add_kernel_size(3);
  pooling_param->add_kernel_size(3);

  pooling_param->add_dilation(2);
  pooling_param->add_dilation(2);
  pooling_param->add_dilation(2);

  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);


  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(1, this->blob_top_->shape(2));
  EXPECT_EQ(1, this->blob_top_->shape(3));
  EXPECT_EQ(1, this->blob_top_->shape(4));
}

TYPED_TEST(PoolingNDSKLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(PoolingNDSKLayerTest, TestBackward) {
  this->TestBackward();
}

}  // namespace caffe
#endif  // !CPU_ONLY
