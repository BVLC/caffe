#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/embed_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class EmbedLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  EmbedLayerTest()
      : blob_bottom_(new Blob<Dtype>(4, 1, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~EmbedLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(EmbedLayerTest, TestDtypesAndDevices);

TYPED_TEST(EmbedLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EmbedParameter* embed_param = layer_param.mutable_embed_param();
  embed_param->set_num_output(10);
  embed_param->set_input_dim(5);
  shared_ptr<EmbedLayer<Dtype> > layer(new EmbedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 5);
  EXPECT_EQ(this->blob_top_->shape(0), 4);
  EXPECT_EQ(this->blob_top_->shape(1), 1);
  EXPECT_EQ(this->blob_top_->shape(2), 1);
  EXPECT_EQ(this->blob_top_->shape(3), 1);
  EXPECT_EQ(this->blob_top_->shape(4), 10);
}

TYPED_TEST(EmbedLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EmbedParameter* embed_param = layer_param.mutable_embed_param();
  const int kNumOutput = 10;
  const int kInputDim = 5;
  embed_param->set_num_output(kNumOutput);
  embed_param->set_input_dim(kInputDim);
  embed_param->mutable_weight_filler()->set_type("uniform");
  embed_param->mutable_weight_filler()->set_min(-10);
  embed_param->mutable_weight_filler()->set_max(10);
  embed_param->set_bias_term(false);
  shared_ptr<EmbedLayer<Dtype> > layer(new EmbedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(1, layer->blobs().size());
  vector<int> weight_shape(2);
  weight_shape[0] = kInputDim;
  weight_shape[1] = kNumOutput;
  ASSERT_TRUE(weight_shape == layer->blobs()[0]->shape());
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    this->blob_bottom_->mutable_cpu_data()[i] = caffe_rng_rand() % kInputDim;
  }
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<int> weight_offset(2, 0);
  vector<int> top_offset(5, 0);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    weight_offset[0] = static_cast<int>(this->blob_bottom_->cpu_data()[i]);
    weight_offset[1] = 0;
    top_offset[0] = i;
    top_offset[4] = 0;
    for (int j = 0; j < kNumOutput; ++j) {
      EXPECT_EQ(layer->blobs()[0]->data_at(weight_offset),
                this->blob_top_->data_at(top_offset));
      ++top_offset[4];
      ++weight_offset[1];
    }
  }
}

TYPED_TEST(EmbedLayerTest, TestForwardWithBias) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EmbedParameter* embed_param = layer_param.mutable_embed_param();
  const int kNumOutput = 10;
  const int kInputDim = 5;
  embed_param->set_num_output(kNumOutput);
  embed_param->set_input_dim(kInputDim);
  embed_param->mutable_weight_filler()->set_type("uniform");
  embed_param->mutable_weight_filler()->set_min(-10);
  embed_param->mutable_weight_filler()->set_max(10);
  embed_param->mutable_bias_filler()->CopyFrom(embed_param->weight_filler());
  embed_param->set_bias_term(true);
  shared_ptr<EmbedLayer<Dtype> > layer(new EmbedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(2, layer->blobs().size());
  vector<int> weight_shape(2);
  weight_shape[0] = kInputDim;
  weight_shape[1] = kNumOutput;
  ASSERT_TRUE(weight_shape == layer->blobs()[0]->shape());
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    this->blob_bottom_->mutable_cpu_data()[i] = caffe_rng_rand() % kInputDim;
  }
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<int> bias_offset(1, 0);
  vector<int> weight_offset(2, 0);
  vector<int> top_offset(5, 0);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    weight_offset[0] = static_cast<int>(this->blob_bottom_->cpu_data()[i]);
    weight_offset[1] = 0;
    top_offset[0] = i;
    top_offset[4] = 0;
    bias_offset[0] = 0;
    for (int j = 0; j < kNumOutput; ++j) {
      EXPECT_EQ(layer->blobs()[0]->data_at(weight_offset) +
                layer->blobs()[1]->data_at(bias_offset),
                this->blob_top_->data_at(top_offset));
      ++top_offset[4];
      ++weight_offset[1];
      ++bias_offset[0];
    }
  }
}

TYPED_TEST(EmbedLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EmbedParameter* embed_param = layer_param.mutable_embed_param();
  embed_param->set_num_output(10);
  embed_param->set_input_dim(5);
  embed_param->set_bias_term(false);
  embed_param->mutable_weight_filler()->set_type("uniform");
  embed_param->mutable_weight_filler()->set_min(-10);
  embed_param->mutable_weight_filler()->set_max(10);
  EmbedLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  this->blob_bottom_->mutable_cpu_data()[0] = 4;
  this->blob_bottom_->mutable_cpu_data()[1] = 2;
  this->blob_bottom_->mutable_cpu_data()[2] = 2;
  this->blob_bottom_->mutable_cpu_data()[3] = 3;
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, -2);
}

TYPED_TEST(EmbedLayerTest, TestGradientWithBias) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EmbedParameter* embed_param = layer_param.mutable_embed_param();
  embed_param->set_num_output(10);
  embed_param->set_input_dim(5);
  embed_param->set_bias_term(true);
  embed_param->mutable_weight_filler()->set_type("uniform");
  embed_param->mutable_weight_filler()->set_min(-10);
  embed_param->mutable_weight_filler()->set_max(10);
  embed_param->mutable_bias_filler()->CopyFrom(embed_param->weight_filler());
  EmbedLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  this->blob_bottom_->mutable_cpu_data()[0] = 4;
  this->blob_bottom_->mutable_cpu_data()[1] = 2;
  this->blob_bottom_->mutable_cpu_data()[2] = 2;
  this->blob_bottom_->mutable_cpu_data()[3] = 3;
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, -2);
}

}  // namespace caffe
