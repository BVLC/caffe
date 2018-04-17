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

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/pooling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::min;

namespace caffe {

template <typename TypeParam>
class StochasticPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  StochasticPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0.1);
    filler_param.set_max(1.);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~StochasticPoolingLayerTest() {
    delete blob_bottom_; delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename Dtype>
class CPUStochasticPoolingLayerTest
  : public StochasticPoolingLayerTest<CPUDevice<Dtype> > {
};

TYPED_TEST_CASE(CPUStochasticPoolingLayerTest, TestDtypes);

TYPED_TEST(CPUStochasticPoolingLayerTest, TestSetup) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

#ifndef CPU_ONLY

template <typename Dtype>
class GPUStochasticPoolingLayerTest
  : public StochasticPoolingLayerTest<GPUDevice<Dtype> > {
};

TYPED_TEST_CASE(GPUStochasticPoolingLayerTest, TestDtypes);

TYPED_TEST(GPUStochasticPoolingLayerTest, TestStochastic) {
  LayerParameter layer_param;
  layer_param.set_phase(TRAIN);
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_STOCHASTIC);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Check if the output is correct - it should do random sampling
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  TypeParam total = 0;
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int ph = 0; ph < this->blob_top_->height(); ++ph) {
        for (int pw = 0; pw < this->blob_top_->width(); ++pw) {
          TypeParam pooled = top_data[this->blob_top_->offset(n, c, ph, pw)];
          total += pooled;
          int hstart = ph * 2;
          int hend = min(hstart + 3, this->blob_bottom_->height());
          int wstart = pw * 2;
          int wend = min(wstart + 3, this->blob_bottom_->width());
          bool has_equal = false;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              has_equal |= (pooled == bottom_data[this->blob_bottom_->
                  offset(n, c, h, w)]);
            }
          }
          EXPECT_TRUE(has_equal);
        }
      }
    }
  }
  // When we are doing stochastic pooling, the average we get should be higher
  // than the simple data average since we are weighting more on higher-valued
  // ones.
  EXPECT_GE(total / this->blob_top_->count(), 0.55);
}

TYPED_TEST(GPUStochasticPoolingLayerTest, TestStochasticTestPhase) {
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_STOCHASTIC);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Check if the output is correct - it should do random sampling
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int ph = 0; ph < this->blob_top_->height(); ++ph) {
        for (int pw = 0; pw < this->blob_top_->width(); ++pw) {
          TypeParam pooled = top_data[this->blob_top_->offset(n, c, ph, pw)];
          int hstart = ph * 2;
          int hend = min(hstart + 3, this->blob_bottom_->height());
          int wstart = pw * 2;
          int wend = min(wstart + 3, this->blob_bottom_->width());
          bool smaller_than_max = false;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              smaller_than_max |= (pooled <= bottom_data[this->blob_bottom_->
                  offset(n, c, h, w)]);
            }
          }
          EXPECT_TRUE(smaller_than_max);
        }
      }
    }
  }
}

TYPED_TEST(GPUStochasticPoolingLayerTest, TestGradient) {
  LayerParameter layer_param;
  layer_param.set_phase(TRAIN);
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_STOCHASTIC);
  PoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-4, 1e-2);
  // it is too expensive to call curand multiple times, so we don't do an
  // exhaustive gradient check.
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

#endif

}  // namespace caffe
