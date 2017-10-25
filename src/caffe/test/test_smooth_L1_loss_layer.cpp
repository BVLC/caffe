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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/smooth_L1_loss_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SmoothL1LossLayerTest : public MultiDeviceTest<TypeParam> {
 typedef typename TypeParam::Dtype Dtype;

 protected:
  SmoothL1LossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_inside_weights_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_outside_weights_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter const_filler_param;
    const_filler_param.set_value(-1.);
    ConstantFiller<Dtype> const_filler(const_filler_param);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);

    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);

    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);

    filler.Fill(this->blob_bottom_inside_weights_);
    blob_bottom_vec_.push_back(blob_bottom_inside_weights_);

    filler.Fill(this->blob_bottom_outside_weights_);
    blob_bottom_vec_.push_back(blob_bottom_outside_weights_);

    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SmoothL1LossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_inside_weights_;
    delete blob_bottom_outside_weights_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_inside_weights_;
  Blob<Dtype>* const blob_bottom_outside_weights_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SmoothL1LossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SmoothL1LossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  SmoothL1LossParameter* loss_param =
      layer_param.mutable_smooth_l1_loss_param();
  loss_param->set_sigma(2.4);

  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  SmoothL1LossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}

}  // namespace caffe
