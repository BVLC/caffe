#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifndef CPU_ONLY  // CPU-GPU test

namespace caffe {

template<typename TypeParam>
class ConvolutionNDLayerTest : public GPUDeviceTest<TypeParam> {
 protected:
  ConvolutionNDLayerTest()
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
    // fill the values
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ConvolutionNDLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  void TestForward() {
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();

    convolution_param->add_kernel_size(3);
    convolution_param->add_kernel_size(3);
    convolution_param->add_kernel_size(3);

    convolution_param->add_kstride(2);
    convolution_param->add_kstride(2);
    convolution_param->add_kstride(2);

    convolution_param->set_num_output(4);

    convolution_param->mutable_weight_filler()->set_type("constant");
    convolution_param->mutable_weight_filler()->set_value(1);
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0);

    ConvolutionNDLayer<TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    int d = blob_bottom_->shape(2);
    int h = blob_bottom_->shape(3);
    int w = blob_bottom_->shape(4);

    TypeParam *bottom_data = blob_bottom_->mutable_cpu_data();

    for (int cd = 0; cd < d; ++cd) {
      for (int ch = 0; ch < h; ++ch) {
        for (int cw = 0; cw < w; ++cw) {
          bottom_data[cw + ch * w + cd * w * h] =
              cw + ch * w + cd * w * h;
        }
      }
    }

  }

  void TestBackward() {
  }

  Blob<TypeParam>* const blob_bottom_;
  Blob<TypeParam>* const blob_top_;

  vector<Blob<TypeParam>*> blob_bottom_vec_;
  vector<Blob<TypeParam>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConvolutionNDLayerTest, TestDtypes);

TYPED_TEST(ConvolutionNDLayerTest, TestSetup) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();

  convolution_param->add_kernel_size(3);
  convolution_param->add_kernel_size(3);
  convolution_param->add_kernel_size(3);

  convolution_param->add_kstride(2);
  convolution_param->add_kstride(2);
  convolution_param->add_kstride(2);

  convolution_param->set_num_output(4);


  ConvolutionNDLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(1, this->blob_top_->shape(2));
  EXPECT_EQ(1, this->blob_top_->shape(3));
  EXPECT_EQ(1, this->blob_top_->shape(4));
}

TYPED_TEST(ConvolutionNDLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(ConvolutionNDLayerTest, TestBackward) {
  this->TestBackward();
}

}  // namespace caffe
#endif  // !CPU_ONLY
