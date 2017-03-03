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

#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/batch_reindex_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class BatchReindexLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BatchReindexLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_permute_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
  }
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    vector<int> sz;
    sz.push_back(5);
    sz.push_back(4);
    sz.push_back(3);
    sz.push_back(2);
    blob_bottom_->Reshape(sz);
    vector<int> permsz;
    permsz.push_back(6);
    blob_bottom_permute_->Reshape(permsz);

    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    int perm[] = { 4, 0, 4, 0, 1, 2 };
    for (int i = 0; i < blob_bottom_permute_->count(); ++i) {
      blob_bottom_permute_->mutable_cpu_data()[i] = perm[i];
    }

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_permute_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~BatchReindexLayerTest() {
    delete blob_bottom_permute_;
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_permute_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward() {
    LayerParameter layer_param;

    vector<int> sz;
    sz.push_back(5);
    sz.push_back(4);
    sz.push_back(3);
    sz.push_back(2);
    blob_bottom_->Reshape(sz);
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      blob_bottom_->mutable_cpu_data()[i] = i;
    }

    vector<int> permsz;
    permsz.push_back(6);
    blob_bottom_permute_->Reshape(permsz);
    int perm[] = { 4, 0, 4, 0, 1, 2 };
    for (int i = 0; i < blob_bottom_permute_->count(); ++i) {
      blob_bottom_permute_->mutable_cpu_data()[i] = perm[i];
    }
    BatchReindexLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), blob_bottom_permute_->num());
    EXPECT_EQ(blob_top_->channels(), blob_bottom_->channels());
    EXPECT_EQ(blob_top_->height(), blob_bottom_->height());
    EXPECT_EQ(blob_top_->width(), blob_bottom_->width());

    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    int channels = blob_top_->channels();
    int height = blob_top_->height();
    int width = blob_top_->width();
    for (int i = 0; i < blob_top_->count(); ++i) {
      int n = i / (channels * width * height);
      int inner_idx = (i % (channels * width * height));
      EXPECT_EQ(
          blob_top_->cpu_data()[i],
          blob_bottom_->cpu_data()[perm[n] * channels * width * height
              + inner_idx]);
    }
  }
};

TYPED_TEST_CASE(BatchReindexLayerTest, TestDtypesAndDevices);

TYPED_TEST(BatchReindexLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(BatchReindexLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BatchReindexLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  }

}  // namespace caffe
