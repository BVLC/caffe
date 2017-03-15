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
#include <algorithm>
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
class MKLDNNEltwiseLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MKLDNNEltwiseLayerTest()
      : blob_bottom_a_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_b_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_c_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);
    filler.Fill(this->blob_bottom_c_);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_bottom_vec_.push_back(blob_bottom_c_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MKLDNNEltwiseLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_bottom_c_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_b_;
  Blob<Dtype>* const blob_bottom_c_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<CPUDevice<float> > TestDtypesCPU;
TYPED_TEST_CASE(MKLDNNEltwiseLayerTest, TestDtypesCPU);

TYPED_TEST(MKLDNNEltwiseLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  shared_ptr<MKLDNNEltwiseLayer<Dtype> > layer(
      new MKLDNNEltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

/*
TYPED_TEST(MKLDNNEltwiseLayerTest, TestProd) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
  shared_ptr<MKLDNNEltwiseLayer<Dtype> > layer(
      new MKLDNNEltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  const Dtype* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[i] * in_data_c[i], 1e-4);
  }
}
*/

TYPED_TEST(MKLDNNEltwiseLayerTest, TestSum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  shared_ptr<MKLDNNEltwiseLayer<Dtype> > layer(
      new MKLDNNEltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  //cout << "Debug info: " << "count of top blob: " << count << endl;
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  const Dtype* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    //cout << "Debug info: " << "data[" << i << "]: " << data[i] << endl;
    //cout << "Debug info: " << "in_data_a[" << i << "]: " << in_data_a[i] << endl;
    //cout << "Debug info: " << "in_data_b[" << i << "]: " << in_data_b[i] << endl;
    //cout << "Debug info: " << "in_data_c[" << i << "]: " << in_data_c[i] << endl;
    //cout << "Debug info: " << "sum result: " << in_data_a[i] + in_data_b[i] + in_data_c[i] << endl;
    //cout << "Debug info: " << "End of sum execution of data[" << i << "]" << endl;
    EXPECT_NEAR(data[i], in_data_a[i] + in_data_b[i] + in_data_c[i], 1e-4);
  }
}

TYPED_TEST(MKLDNNEltwiseLayerTest, TestSumCoeff) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  eltwise_param->add_coeff(1);
  eltwise_param->add_coeff(-0.5);
  eltwise_param->add_coeff(2);
  shared_ptr<MKLDNNEltwiseLayer<Dtype> > layer(
      new MKLDNNEltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  const Dtype* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] - 0.5*in_data_b[i] + 2*in_data_c[i],
        1e-4);
  }
}

/*
TYPED_TEST(MKLDNNEltwiseLayerTest, TestStableProdGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
  eltwise_param->set_stable_prod_grad(true);
  MKLDNNEltwiseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MKLDNNEltwiseLayerTest, TestUnstableProdGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
  eltwise_param->set_stable_prod_grad(false);
  MKLDNNEltwiseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
*/
TYPED_TEST(MKLDNNEltwiseLayerTest, TestSumGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  MKLDNNEltwiseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
/*
TYPED_TEST(MKLDNNEltwiseLayerTest, TestSumCoeffGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  eltwise_param->add_coeff(1);
  eltwise_param->add_coeff(-0.5);
  eltwise_param->add_coeff(2);
  MKLDNNEltwiseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MKLDNNEltwiseLayerTest, TestMax) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_MAX);
  shared_ptr<MKLDNNEltwiseLayer<Dtype> > layer(
      new MKLDNNEltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  const Dtype* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(data[i],
              std::max(in_data_a[i], std::max(in_data_b[i], in_data_c[i])));
  }
}

TYPED_TEST(MKLDNNEltwiseLayerTest, TestMaxGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_MAX);
  MKLDNNEltwiseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
*/
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
