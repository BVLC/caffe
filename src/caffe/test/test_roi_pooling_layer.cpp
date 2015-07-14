// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/fast_rcnn_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

typedef ::testing::Types<GPUDevice<float>, GPUDevice<double> > TestDtypesGPU;

template <typename TypeParam>
class ROIPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ROIPoolingLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(4, 3, 12, 8)),
        blob_bottom_rois_(new Blob<Dtype>(4, 5, 1, 1)),
        blob_top_data_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    //for (int i = 0; i < blob_bottom_data_->count(); ++i) {
    //  blob_bottom_data_->mutable_cpu_data()[i] = i;
    //}
    blob_bottom_vec_.push_back(blob_bottom_data_);
    int i = 0;
    blob_bottom_rois_->mutable_cpu_data()[0 + 5*i] = 0; //caffe_rng_rand() % 4;
    blob_bottom_rois_->mutable_cpu_data()[1 + 5*i] = 1; // x1 < 8
    blob_bottom_rois_->mutable_cpu_data()[2 + 5*i] = 1; // y1 < 12
    blob_bottom_rois_->mutable_cpu_data()[3 + 5*i] = 6; // x2 < 8
    blob_bottom_rois_->mutable_cpu_data()[4 + 5*i] = 6; // y2 < 12
    i = 1;
    blob_bottom_rois_->mutable_cpu_data()[0 + 5*i] = 2;
    blob_bottom_rois_->mutable_cpu_data()[1 + 5*i] = 6; // x1 < 8
    blob_bottom_rois_->mutable_cpu_data()[2 + 5*i] = 2; // y1 < 12
    blob_bottom_rois_->mutable_cpu_data()[3 + 5*i] = 7; // x2 < 8
    blob_bottom_rois_->mutable_cpu_data()[4 + 5*i] = 11; // y2 < 12
    i = 2;
    blob_bottom_rois_->mutable_cpu_data()[0 + 5*i] = 1;
    blob_bottom_rois_->mutable_cpu_data()[1 + 5*i] = 3; // x1 < 8
    blob_bottom_rois_->mutable_cpu_data()[2 + 5*i] = 1; // y1 < 12
    blob_bottom_rois_->mutable_cpu_data()[3 + 5*i] = 5; // x2 < 8
    blob_bottom_rois_->mutable_cpu_data()[4 + 5*i] = 10; // y2 < 12
    i = 3;
    blob_bottom_rois_->mutable_cpu_data()[0 + 5*i] = 0;
    blob_bottom_rois_->mutable_cpu_data()[1 + 5*i] = 3; // x1 < 8
    blob_bottom_rois_->mutable_cpu_data()[2 + 5*i] = 3; // y1 < 12
    blob_bottom_rois_->mutable_cpu_data()[3 + 5*i] = 3; // x2 < 8
    blob_bottom_rois_->mutable_cpu_data()[4 + 5*i] = 3; // y2 < 12

    blob_bottom_vec_.push_back(blob_bottom_rois_);
    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~ROIPoolingLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_rois_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_rois_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ROIPoolingLayerTest, TestDtypesGPU);

TYPED_TEST(ROIPoolingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ROIPoolingParameter* roi_pooling_param =
      layer_param.mutable_roi_pooling_param();
  roi_pooling_param->set_pooled_h(6);
  roi_pooling_param->set_pooled_w(6);
  ROIPoolingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
