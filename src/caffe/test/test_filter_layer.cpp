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
#include "caffe/layers/filter_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class FilterLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  FilterLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(4, 3, 6, 4)),
        blob_bottom_labels_(new Blob<Dtype>(4, 1, 1, 1)),
        blob_bottom_selector_(new Blob<Dtype>(4, 1, 1, 1)),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_labels_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    Caffe::set_random_seed(1890);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    // fill the selector blob
    Dtype* bottom_data_selector_ = blob_bottom_selector_->mutable_cpu_data();
    bottom_data_selector_[0] = 0;
    bottom_data_selector_[1] = 1;
    bottom_data_selector_[2] = 1;
    bottom_data_selector_[3] = 0;
    // fill the other bottom blobs
    filler.Fill(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_labels_->count(); ++i) {
      blob_bottom_labels_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_labels_);
    blob_bottom_vec_.push_back(blob_bottom_selector_);
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_labels_);
  }
  virtual ~FilterLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_labels_;
    delete blob_bottom_selector_;
    delete blob_top_data_;
    delete blob_top_labels_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_labels_;
  Blob<Dtype>* const blob_bottom_selector_;
  // blobs for the top of FilterLayer
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_labels_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(FilterLayerTest, TestDtypesAndDevices);

TYPED_TEST(FilterLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FilterLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  // In the test first and last items should have been filtered
  // so we just expect 2 remaining items
  EXPECT_EQ(this->blob_top_data_->shape(0), 2);
  EXPECT_EQ(this->blob_top_labels_->shape(0), 2);
  EXPECT_GT(this->blob_bottom_data_->shape(0),
      this->blob_top_data_->shape(0));
  EXPECT_GT(this->blob_bottom_labels_->shape(0),
      this->blob_top_labels_->shape(0));
  for (int i = 1; i < this->blob_bottom_labels_->num_axes(); i++) {
    EXPECT_EQ(this->blob_bottom_labels_->shape(i),
        this->blob_top_labels_->shape(i));
  }
}

TYPED_TEST(FilterLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FilterLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_labels_->data_at(0, 0, 0, 0),
      this->blob_bottom_labels_->data_at(1, 0, 0, 0));
  EXPECT_EQ(this->blob_top_labels_->data_at(1, 0, 0, 0),
      this->blob_bottom_labels_->data_at(2, 0, 0, 0));

  int dim = this->blob_top_data_->count() /
      this->blob_top_data_->shape(0);
  const Dtype* top_data = this->blob_top_data_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_data_->cpu_data();
  // selector is 0 1 1 0, so we need to compare bottom(1,c,h,w)
  // with top(0,c,h,w) and bottom(2,c,h,w) with top(1,c,h,w)
  bottom_data += dim;  // bottom(1,c,h,w)
  for (size_t n = 0; n < dim; n++)
    EXPECT_EQ(top_data[n], bottom_data[n]);

  bottom_data += dim;  // bottom(2,c,h,w)
  top_data += dim;  // top(1,c,h,w)
  for (size_t n = 0; n < dim; n++)
    EXPECT_EQ(top_data[n], bottom_data[n]);
}

TYPED_TEST(FilterLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FilterLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  // check only input 0 (data) because labels and selector
  // don't need backpropagation
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
