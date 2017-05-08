/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef MKLDNN_SUPPORTED
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mkldnn_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MKLDNNInnerProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MKLDNNInnerProductLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_nobatch_(new Blob<Dtype>(1, 2, 3, 4)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MKLDNNInnerProductLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_nobatch_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_nobatch_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<CPUDevice<float>
//                        ,CPUDevice<double>
                        > TestDtypesCPU;

TYPED_TEST_CASE(MKLDNNInnerProductLayerTest, TestDtypesCPU);

TYPED_TEST(MKLDNNInnerProductLayerTest, TestSetupMKLDNN) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new MKLDNNInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 10);
}

/** @brief TestSetUp while toggling tranpose flag
 */
TYPED_TEST(MKLDNNInnerProductLayerTest, TestSetUpTranposeFalse) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->set_transpose(false);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new MKLDNNInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(2, this->blob_top_->num());
  EXPECT_EQ(1, this->blob_top_->height());
  EXPECT_EQ(1, this->blob_top_->width());
  EXPECT_EQ(10, this->blob_top_->channels());
  EXPECT_EQ(2, layer->blobs()[0]->num_axes());
  EXPECT_EQ(10, layer->blobs()[0]->shape(0));
  EXPECT_EQ(60, layer->blobs()[0]->shape(1));
}

/** @brief TestSetUp while toggling tranpose flag
 */
TYPED_TEST(MKLDNNInnerProductLayerTest, TestSetUpTranposeTrue) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->set_transpose(true);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new MKLDNNInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(2, this->blob_top_->num());
  EXPECT_EQ(1, this->blob_top_->height());
  EXPECT_EQ(1, this->blob_top_->width());
  EXPECT_EQ(10, this->blob_top_->channels());
  EXPECT_EQ(2, layer->blobs()[0]->num_axes());
  EXPECT_EQ(60, layer->blobs()[0]->shape(0));
  EXPECT_EQ(10, layer->blobs()[0]->shape(1));
}

TYPED_TEST(MKLDNNInnerProductLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new MKLDNNInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], 1.);
  }
}

TYPED_TEST(MKLDNNInnerProductLayerTest, TestForwardNoBias) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->set_bias_term(false);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new MKLDNNInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], 1.);
  }
}

// TODO: add support for transposed weights in MKLDNNInnerProduct 
// layer and then enable following test (check if it was ported properly)
#if 0
/**
 * @brief Init. an IP layer without transpose + random weights,
 * run Forward, save the result.
 * Init. another IP layer with transpose.
 * manually copy and transpose the weights from the first IP layer,
 * then run Forward on the same input and check that the result is the same
 */
TYPED_TEST(MKLDNNInnerProductLayerTest, TestForwardTranspose) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  inner_product_param->set_transpose(false);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new MKLDNNInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const int count = this->blob_top_->count();
  Blob<Dtype>* const top = new Blob<Dtype>();
  top->ReshapeLike(*this->blob_top_);
  caffe_copy(count, this->blob_top_->cpu_data(), top->mutable_cpu_data());
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(new Blob<Dtype>());
  inner_product_param->set_transpose(true);
  shared_ptr<InnerProductLayer<Dtype> > ip_t(
      new MKLDNNInnerProductLayer<Dtype>(layer_param));
  ip_t->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const int count_w = layer->blobs()[0]->count();
  EXPECT_EQ(count_w, ip_t->blobs()[0]->count());
  // manually copy and transpose the weights from 1st IP layer into 2nd
  const Dtype* w = layer->blobs()[0]->cpu_data();
  Dtype* w_t = ip_t->blobs()[0]->mutable_cpu_data();
  const int width = layer->blobs()[0]->shape(1);
  const int width_t = ip_t->blobs()[0]->shape(1);
  for (int i = 0; i < count_w; ++i) {
    int r = i / width;
    int c = i % width;
    w_t[c*width_t+r] = w[r*width+c];  // copy while transposing
  }
  // copy bias from 1st IP layer to 2nd IP layer
  ASSERT_EQ(layer->blobs()[1]->count(), ip_t->blobs()[1]->count());
  caffe_copy(layer->blobs()[1]->count(), layer->blobs()[1]->cpu_data(),
      ip_t->blobs()[1]->mutable_cpu_data());
  ip_t->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(count, this->blob_top_->count())
      << "Invalid count for top blob for IP with transpose.";
  Blob<Dtype>* const top_t = new Blob<Dtype>();\
  top_t->ReshapeLike(*this->blob_top_vec_[0]);
  caffe_copy(count,
    this->blob_top_vec_[0]->cpu_data(),
    top_t->mutable_cpu_data());
  const Dtype* data = top->cpu_data();
  const Dtype* data_t = top_t->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(data[i], data_t[i]);
  }
}
#endif

TYPED_TEST(MKLDNNInnerProductLayerTest, TestForwardNoBatch) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_nobatch_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new MKLDNNInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], 1.);
  }
}

TYPED_TEST(MKLDNNInnerProductLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    shared_ptr<InnerProductLayer<Dtype> > layer(
      new MKLDNNInnerProductLayer<Dtype>(layer_param));
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_,
        this->blob_top_vec_);
}

TYPED_TEST(MKLDNNInnerProductLayerTest, TestGradientNoBias) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    inner_product_param->set_bias_term(false);
    shared_ptr<InnerProductLayer<Dtype> > layer(
      new MKLDNNInnerProductLayer<Dtype>(layer_param));
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_,
        this->blob_top_vec_);
}

TYPED_TEST(MKLDNNInnerProductLayerTest, TestGradientTranspose) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(11);
  inner_product_param->mutable_weight_filler()->set_type("gaussian");
  inner_product_param->mutable_bias_filler()->set_type("gaussian");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  inner_product_param->set_transpose(true);
  shared_ptr<InnerProductLayer<Dtype> > layer(
    new MKLDNNInnerProductLayer<Dtype>(layer_param));
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
