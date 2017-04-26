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

#if defined(MKLDNN_SUPPORTED)
#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/mkldnn_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define MB 2
#define IC 4
#define IH 5
#define IW 5
#define LS 3


namespace caffe {

  template <typename TypeParam>
  class MKLDNNBatchNormLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
   protected:
    MKLDNNBatchNormLayerTest()
        : blob_bottom_(new Blob<Dtype>(MB, IC, IH, IW)),
          blob_top_(new Blob<Dtype>()) {
      // fill the values
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~MKLDNNBatchNormLayerTest() {delete blob_bottom_; delete blob_top_;}
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
  };

  typedef ::testing::Types<CPUDevice<float>/*,
                         CPUDevice<double>*/ > TestDtypesCPU;
  TYPED_TEST_CASE(MKLDNNBatchNormLayerTest, TestDtypesCPU);

  TYPED_TEST(MKLDNNBatchNormLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    MKLDNNBatchNormLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Test mean
    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();

    for (int j = 0; j < channels; ++j) {
      Dtype sum = 0, var = 0;
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = this->blob_top_->data_at(i, j, k, l);
            sum += data;
            var += data * data;
          }
        }
      }
      sum /= height * width * num;
      var /= height * width * num;

      const Dtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
    }
  }

  TYPED_TEST(MKLDNNBatchNormLayerTest, TestForwardInplace) {
    typedef typename TypeParam::Dtype Dtype;
    Blob<Dtype> blob_inplace(MB, IC, IH, IW);
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    LayerParameter layer_param;
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_inplace);
    blob_bottom_vec.push_back(&blob_inplace);
    blob_top_vec.push_back(&blob_inplace);

    MKLDNNBatchNormLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec, blob_top_vec);
    layer.Forward(blob_bottom_vec, blob_top_vec);

    // Test mean
    int num = blob_inplace.num();
    int channels = blob_inplace.channels();
    int height = blob_inplace.height();
    int width = blob_inplace.width();

    for (int j = 0; j < channels; ++j) {
      Dtype sum = 0, var = 0;
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = blob_inplace.data_at(i, j, k, l);
            sum += data;
            var += data * data;
          }
        }
      }
      sum /= height * width * num;
      var /= height * width * num;

      const Dtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
    }
  }

  TYPED_TEST(MKLDNNBatchNormLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    MKLDNNBatchNormLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }


}  // namespace caffe
#endif  // #if defined(MKLDNN_SUPPORTED)
