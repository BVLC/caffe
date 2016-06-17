#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/attention_layers.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SoftAttentionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftAttentionLayerTest() {
    blob_bottom_a_.Reshape(3, 4, 5, 6);
    blob_bottom_alpha_.Reshape(3, 1, 5, 6);
    blob_bottom_beta_.Reshape(3, 1, 1, 1);
    blob_ref_top_z_.Reshape(3, 4, 1, 1);
    FillerParameter filler_param;
    filler_param.set_std(0.5);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_bottom_a_);
    filler.Fill(&blob_bottom_alpha_);
    filler.Fill(&blob_bottom_beta_);
    blob_bottom_vec_.push_back(&blob_bottom_a_);
    blob_bottom_vec_.push_back(&blob_bottom_alpha_);
    blob_bottom_vec_.push_back(&blob_bottom_beta_);
    blob_top_vec_.push_back(&blob_top_z_);

    int num = 3, channels = 4, spatial_dim = 5 * 6;
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        blob_ref_top_z_.mutable_cpu_data()[n*channels+c] =
            blob_bottom_beta_.cpu_data()[n] * caffe_cpu_dot(
            spatial_dim, blob_bottom_alpha_.cpu_data() + n * spatial_dim,
            blob_bottom_a_.cpu_data() + (n * channels + c) * spatial_dim);
      }
    }
  }

  Blob<Dtype> blob_bottom_a_;
  Blob<Dtype> blob_bottom_alpha_;
  Blob<Dtype> blob_bottom_beta_;
  Blob<Dtype> blob_top_z_;
  Blob<Dtype> blob_ref_top_z_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftAttentionLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftAttentionLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  SoftAttentionLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_z_.num_axes(), 4);
  EXPECT_EQ(this->blob_top_z_.num(), 3);
  EXPECT_EQ(this->blob_top_z_.channels(), 4);
  EXPECT_EQ(this->blob_top_z_.height(), 1);
  EXPECT_EQ(this->blob_top_z_.width(), 1);
}

TYPED_TEST(SoftAttentionLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  SoftAttentionLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SoftAttentionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  SoftAttentionLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
