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
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/detection_evaluate_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

static const float eps = 1e-6;

template <typename Dtype>
class DetectionEvaluateLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  DetectionEvaluateLayerTest()
      : num_classes_(3),
        background_label_id_(0),
        overlap_threshold_(0.3),
        blob_bottom_det_(new Blob<Dtype>(1, 1, 8, 7)),
        blob_bottom_gt_(new Blob<Dtype>(1, 1, 4, 8)),
        blob_top_(new Blob<Dtype>()) {
    this->FillData();
    blob_bottom_vec_.push_back(blob_bottom_det_);
    blob_bottom_vec_.push_back(blob_bottom_gt_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~DetectionEvaluateLayerTest() {
    delete blob_bottom_det_;
    delete blob_bottom_gt_;
    delete blob_top_;
  }

  void FillData() {
    // Fill ground truth.
    bool is_gt = true;
    FillItem(blob_bottom_gt_, 0, "0 1 0 0.1 0.1 0.3 0.3 0", is_gt);
    FillItem(blob_bottom_gt_, 1, "0 1 0 0.6 0.6 0.8 0.8 1", is_gt);
    FillItem(blob_bottom_gt_, 2, "1 2 0 0.3 0.3 0.6 0.5 0", is_gt);
    FillItem(blob_bottom_gt_, 3, "1 1 0 0.7 0.1 0.9 0.3 0", is_gt);

    // Fill detections.
    is_gt = false;
    FillItem(blob_bottom_det_, 0, "0 1 0.3 0.1 0.0 0.4 0.3", is_gt);
    FillItem(blob_bottom_det_, 1, "0 1 0.7 0.0 0.1 0.2 0.3", is_gt);
    FillItem(blob_bottom_det_, 2, "0 1 0.9 0.7 0.6 0.8 0.8", is_gt);
    FillItem(blob_bottom_det_, 3, "1 2 0.8 0.2 0.1 0.4 0.4", is_gt);
    FillItem(blob_bottom_det_, 4, "1 2 0.1 0.4 0.3 0.7 0.5", is_gt);
    FillItem(blob_bottom_det_, 5, "1 1 0.2 0.8 0.1 1.0 0.3", is_gt);
    FillItem(blob_bottom_det_, 6, "1 3 0.2 0.8 0.1 1.0 0.3", is_gt);
    FillItem(blob_bottom_det_, 7, "2 1 0.2 0.8 0.1 1.0 0.3", is_gt);
  }

  void FillItem(Blob<Dtype>* blob, const int item, const string values,
                const bool is_gt) {
    CHECK_LT(item, blob->height());

    // Split values to vector of items.
    vector<string> items;
    std::istringstream iss(values);
    std::copy(std::istream_iterator<string>(iss),
              std::istream_iterator<string>(), back_inserter(items));
    if (is_gt) {
      EXPECT_EQ(items.size(), 8);
    } else {
      EXPECT_EQ(items.size(), 7);
    }
    int num_items = items.size();

    // Fill item.
    Dtype* blob_data = blob->mutable_cpu_data();
    for (int i = 0; i < 2; ++i) {
      blob_data[item * num_items + i] = atoi(items[i].c_str());
    }
    for (int i = 2; i < 7; ++i) {
      blob_data[item * num_items + i] = atof(items[i].c_str());
    }
    if (is_gt) {
      blob_data[item * num_items + 7] = atoi(items[7].c_str());
    }
  }

  void CheckEqual(const Blob<Dtype>& blob, const int num, const string values) {
    CHECK_LT(num, blob.height());

    // Split values to vector of items.
    vector<string> items;
    std::istringstream iss(values);
    std::copy(std::istream_iterator<string>(iss),
              std::istream_iterator<string>(), back_inserter(items));
    EXPECT_EQ(items.size(), 5);

    // Check data.
    const Dtype* blob_data = blob.cpu_data();
    for (int i = 0; i < 5; ++i) {
      if (i == 2) {
        EXPECT_NEAR(blob_data[num * blob.width() + i],
                    atof(items[i].c_str()), eps);
      } else {
        EXPECT_EQ(static_cast<int>(blob_data[num * blob.width() + i]),
                  atoi(items[i].c_str()));
      }
    }
  }

  int num_classes_;
  int background_label_id_;
  float overlap_threshold_;

  Blob<Dtype>* const blob_bottom_det_;
  Blob<Dtype>* const blob_bottom_gt_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DetectionEvaluateLayerTest, TestDtypes);

TYPED_TEST(DetectionEvaluateLayerTest, TestSetup) {
  LayerParameter layer_param;
  DetectionEvaluateParameter* detection_evaluate_param =
      layer_param.mutable_detection_evaluate_param();
  detection_evaluate_param->set_num_classes(this->num_classes_);
  detection_evaluate_param->set_background_label_id(this->background_label_id_);
  detection_evaluate_param->set_overlap_threshold(this->overlap_threshold_);
  DetectionEvaluateLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_det_->height() + 2);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(DetectionEvaluateLayerTest, TestForward) {
  LayerParameter layer_param;
  DetectionEvaluateParameter* detection_evaluate_param =
      layer_param.mutable_detection_evaluate_param();
  detection_evaluate_param->set_num_classes(this->num_classes_);
  detection_evaluate_param->set_background_label_id(this->background_label_id_);
  detection_evaluate_param->set_overlap_threshold(this->overlap_threshold_);
  DetectionEvaluateLayer<TypeParam> layer(layer_param);

  this->FillData();
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_det_->height() + 2);
  EXPECT_EQ(this->blob_top_->width(), 5);

  this->CheckEqual(*(this->blob_top_), 0, "-1 1 3 -1 -1");
  this->CheckEqual(*(this->blob_top_), 1, "-1 2 1 -1 -1");
  this->CheckEqual(*(this->blob_top_), 2, "0 1 0.9 1 0");
  this->CheckEqual(*(this->blob_top_), 3, "0 1 0.7 1 0");
  this->CheckEqual(*(this->blob_top_), 4, "0 1 0.3 0 1");
  this->CheckEqual(*(this->blob_top_), 5, "1 1 0.2 1 0");
  this->CheckEqual(*(this->blob_top_), 6, "1 2 0.8 0 1");
  this->CheckEqual(*(this->blob_top_), 7, "1 2 0.1 1 0");
  this->CheckEqual(*(this->blob_top_), 8, "1 3 0.2 0 1");
  this->CheckEqual(*(this->blob_top_), 9, "2 1 0.2 0 1");
}

TYPED_TEST(DetectionEvaluateLayerTest, TestForwardSkipDifficult) {
  LayerParameter layer_param;
  DetectionEvaluateParameter* detection_evaluate_param =
      layer_param.mutable_detection_evaluate_param();
  detection_evaluate_param->set_num_classes(this->num_classes_);
  detection_evaluate_param->set_background_label_id(this->background_label_id_);
  detection_evaluate_param->set_overlap_threshold(this->overlap_threshold_);
  detection_evaluate_param->set_evaluate_difficult_gt(false);
  DetectionEvaluateLayer<TypeParam> layer(layer_param);

  this->FillData();
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_det_->height() + 2);
  EXPECT_EQ(this->blob_top_->width(), 5);

  this->CheckEqual(*(this->blob_top_), 0, "-1 1 2 -1 -1");
  this->CheckEqual(*(this->blob_top_), 1, "-1 2 1 -1 -1");
  this->CheckEqual(*(this->blob_top_), 2, "0 1 0.9 0 0");
  this->CheckEqual(*(this->blob_top_), 3, "0 1 0.7 1 0");
  this->CheckEqual(*(this->blob_top_), 4, "0 1 0.3 0 1");
  this->CheckEqual(*(this->blob_top_), 5, "1 1 0.2 1 0");
  this->CheckEqual(*(this->blob_top_), 6, "1 2 0.8 0 1");
  this->CheckEqual(*(this->blob_top_), 7, "1 2 0.1 1 0");
  this->CheckEqual(*(this->blob_top_), 8, "1 3 0.2 0 1");
  this->CheckEqual(*(this->blob_top_), 9, "2 1 0.2 0 1");
}

}  // namespace caffe
