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

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/box_annotator_ohem_layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

  template <typename Dtype>
  void BoxAnnotatorOHEMLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    BoxAnnotatorOHEMParameter box_anno_param =
      this->layer_param_.box_annotator_ohem_param();
    roi_per_img_ = box_anno_param.roi_per_img();
    CHECK_GT(roi_per_img_, 0);
    ignore_label_ = box_anno_param.ignore_label();
  }

  template <typename Dtype>
  void BoxAnnotatorOHEMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    num_ = bottom[0]->num();
    CHECK_EQ(5, bottom[0]->channels());
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    spatial_dim_ = height_*width_;

    CHECK_EQ(bottom[1]->num(), num_);
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->height(), height_);
    CHECK_EQ(bottom[1]->width(), width_);

    CHECK_EQ(bottom[2]->num(), num_);
    CHECK_EQ(bottom[2]->channels(), 1);
    CHECK_EQ(bottom[2]->height(), height_);
    CHECK_EQ(bottom[2]->width(), width_);

    CHECK_EQ(bottom[3]->num(), num_);
    bbox_channels_ = bottom[3]->channels();
    CHECK_EQ(bottom[3]->height(), height_);
    CHECK_EQ(bottom[3]->width(), width_);

    // Labels for scoring
    top[0]->Reshape(num_, 1, height_, width_);
    // Loss weights for bbox regression
    top[1]->Reshape(num_, bbox_channels_, height_, width_);
  }

  template <typename Dtype>
  void BoxAnnotatorOHEMLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        const Dtype* bottom_rois = bottom[0]->cpu_data();
    const Dtype* bottom_loss = bottom[1]->cpu_data();
    const Dtype* bottom_labels = bottom[2]->cpu_data();
    const Dtype* bottom_bbox_loss_weights = bottom[3]->cpu_data();
    Dtype* top_labels = top[0]->mutable_cpu_data();
    Dtype* top_bbox_loss_weights = top[1]->mutable_cpu_data();
    caffe_set(top[0]->count(), Dtype(ignore_label_), top_labels);
    caffe_set(top[1]->count(), Dtype(0), top_bbox_loss_weights);

    int num_rois_ = bottom[1]->count();

    int num_imgs = -1;
    for (int n = 0; n < num_rois_; n++) {
      for (int s = 0; s < spatial_dim_; s++) {
        num_imgs = bottom_rois[0] > num_imgs ? bottom_rois[0] : num_imgs;
        bottom_rois++;
      }
      bottom_rois += (5 - 1) * spatial_dim_;
    }
    num_imgs++;
    CHECK_GT(num_imgs, 0)
      << "number of images must be greater than 0 at BoxAnnotatorOHEMLayer";
    bottom_rois = bottom[0]->cpu_data();

    // Find rois with max loss
    vector<int> sorted_idx(num_rois_);
    for (int i = 0; i < num_rois_; i++) {
      sorted_idx[i] = i;
    }
    std::sort(sorted_idx.begin(), sorted_idx.end(),
      [bottom_loss](int i1, int i2) {
        return bottom_loss[i1] > bottom_loss[i2];
    });

    // Generate output labels for scoring and loss_weights for bbox regression
    vector<int> number_left(num_imgs, roi_per_img_);
    for (int i = 0; i < num_rois_; i++) {
      int index = sorted_idx[i];
      int s = index % (width_ * height_);
      int n = index / (width_ * height_);
      int batch_ind = bottom_rois[n * 5 * spatial_dim_ + s];
      if (number_left[batch_ind] > 0) {
        number_left[batch_ind]--;
        top_labels[index] = bottom_labels[index];
        for (int j = 0; j < bbox_channels_; j++) {
          int bbox_index = (n * bbox_channels_ + j) * spatial_dim_ + s;
          top_bbox_loss_weights[bbox_index] = bottom_bbox_loss_weights[bbox_index];
        }
      }
    }
  }

  template <typename Dtype>
  void BoxAnnotatorOHEMLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    return;
  }


#ifdef CPU_ONLY
  STUB_GPU(BoxAnnotatorOHEMLayer);
#endif

  INSTANTIATE_CLASS(BoxAnnotatorOHEMLayer);
  REGISTER_LAYER_CLASS(BoxAnnotatorOHEM);

}  // namespace caffe
