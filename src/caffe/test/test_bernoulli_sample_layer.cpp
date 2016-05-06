#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/bernoulli_sample_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class BernoulliSampleLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BernoulliSampleLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 2, 2, 2)),
      blob_top_(new Blob<Dtype>()) {
    vect_bottom_.push_back(blob_bottom_);
    vect_top_.push_back(blob_top_);
  }

  virtual ~BernoulliSampleLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  vector<Blob<Dtype>*> vect_bottom_;
  vector<Blob<Dtype>*> vect_top_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
};

TYPED_TEST_CASE(BernoulliSampleLayerTest, TestDtypesAndDevices);

TYPED_TEST(BernoulliSampleLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<BernoulliSampleLayer<Dtype> > layer(
      new BernoulliSampleLayer<Dtype>(layer_param));
  layer->SetUp(this->vect_bottom_, this->vect_top_);
  ASSERT_EQ(4, this->blob_top_->num_axes());
  ASSERT_EQ(2, this->blob_top_->shape(0));
  ASSERT_EQ(2, this->blob_top_->shape(1));
  ASSERT_EQ(2, this->blob_top_->shape(2));
  ASSERT_EQ(2, this->blob_top_->shape(3));
}

TYPED_TEST(BernoulliSampleLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_random_seed(1701);
  const int count = this->blob_bottom_->count();
  caffe_rng_uniform(count,
                    Dtype(0),
                    Dtype(1),
                    this->blob_bottom_->mutable_cpu_data());
  LayerParameter layer_param;
  shared_ptr<BernoulliSampleLayer<Dtype> > layer(
      new BernoulliSampleLayer<Dtype>(layer_param));
  layer->SetUp(this->vect_bottom_, this->vect_top_);
  layer->Forward(this->vect_bottom_, this->vect_top_);
  const Dtype* samp_data = this->blob_top_->cpu_data();

  ASSERT_EQ(this->blob_top_->count(), count);
  // make sure the values are either zero or one
  for (int i = 0; i < count; ++i) {
    EXPECT_TRUE((samp_data[i] == 0 || samp_data[i] == 1));
  }

  // make sure we are close to the actual probability
  const int N = 1000;
  Blob<Dtype> average_blob(this->blob_bottom_->shape());
  average_blob.scale_data(0);
  Dtype* average_data = average_blob.mutable_cpu_data();
  for (int i = 0; i < N; ++i) {
    layer->Forward(this->vect_bottom_, this->vect_top_);
    caffe_axpy(count, Dtype(1.) / N, this->blob_top_->cpu_data(), average_data);
  }

  const Dtype eps = 0.06;
  for (int i = 0; i < count; ++i) {
    EXPECT_LE(this->blob_bottom_->cpu_data()[i] - eps, average_data[i]);
    EXPECT_GE(this->blob_bottom_->cpu_data()[i] + eps, average_data[i]);
  }
}

}  // namespace caffe
