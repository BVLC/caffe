#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffeine/blob.hpp"
#include "caffeine/common.hpp"
#include "caffeine/filler.hpp"
#include "caffeine/vision_layers.hpp"

extern cudaDeviceProp CAFFEINE_TEST_CUDA_PROP;

namespace caffeine {
  
template <typename Dtype>
class InnerProductLayerTest : public ::testing::Test {
 protected:
  InnerProductLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  };
  virtual ~InnerProductLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(InnerProductLayerTest, Dtypes);

TYPED_TEST(InnerProductLayerTest, TestSetUp) {
  LayerParameter layer_param;
  layer_param.set_num_output(10);
  shared_ptr<InnerProductLayer<TypeParam> > layer(
  	new InnerProductLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 10);
  layer_param.set_gemm_last_dim(true);
  layer.reset(new InnerProductLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
  EXPECT_EQ(this->blob_top_->channels(), 10);
}

TYPED_TEST(InnerProductLayerTest, TestCPU) {
  LayerParameter layer_param;
  Caffeine::set_mode(Caffeine::CPU);
  layer_param.set_num_output(10);
  layer_param.mutable_weight_filler()->set_type("uniform");
  layer_param.mutable_bias_filler()->set_type("uniform");
  layer_param.mutable_bias_filler()->set_min(1);
  layer_param.mutable_bias_filler()->set_max(2);
  shared_ptr<InnerProductLayer<TypeParam> > layer(
  	new InnerProductLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  for (int i = 0; i < count; ++i) {
  	EXPECT_GE(data[i], 1.);
  }
}

TYPED_TEST(InnerProductLayerTest, TestGPU) {
	if (sizeof(TypeParam) == 4 || CAFFEINE_TEST_CUDA_PROP.major >= 2) {
	  LayerParameter layer_param;
	  Caffeine::set_mode(Caffeine::GPU);
	  layer_param.set_num_output(10);
	  layer_param.mutable_weight_filler()->set_type("uniform");
	  layer_param.mutable_bias_filler()->set_type("uniform");
	  layer_param.mutable_bias_filler()->set_min(1);
	  layer_param.mutable_bias_filler()->set_max(2);
	  shared_ptr<InnerProductLayer<TypeParam> > layer(
	  	new InnerProductLayer<TypeParam>(layer_param));
	  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
	  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
	  const TypeParam* data = this->blob_top_->cpu_data();
	  const int count = this->blob_top_->count();
	  for (int i = 0; i < count; ++i) {
	  	EXPECT_GE(data[i], 1.);
	  }
	} else {
		LOG(ERROR) << "Skipping test due to old architecture.";
	}
}

}
