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



#ifndef CAFFE_PSROI_POOLING_LAYER_HPP_
#define CAFFE_PSROI_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/** 
 * @brief Perform position-sensitive max pooling on regions of interest specified by input, takes
 *        as input N position-sensitive score maps and a list of R regions of interest.
 *   ROIPoolingLayer takes 2 inputs and produces 1 output. bottom[0] is
 *   [N x (C x K^2) x H x W] position-sensitive score maps on which pooling is performed. bottom[1] is
 *   [R x 5] containing a list R ROI tuples with batch index and coordinates of
 *   regions of interest. Each row in bottom[1] is a ROI tuple in format
 *   [batch_index x1 y1 x2 y2], where batch_index corresponds to the index of
 *   instance in the first input and x1 y1 x2 y2 are 0-indexed coordinates
 *   of ROI rectangle (including its boundaries). The output top[0] is [R x C x K x K] score maps pooled
 *   within the ROI tuples.
 * @param param provides PSROIPoolingParameter psroi_pooling_param,
 *        with PSROIPoolingLayer options:
 *  - output_dim. The pooled output channel number.
 *  - group_size. The number of groups to encode position-sensitive score maps
 *  - spatial_scale. Multiplicative spatial scale factor to translate ROI
 *  coordinates from their input scale to the scale used when pooling.
 * R-FCN
 * Written by Yi Li
 */

template <typename Dtype>
class PSROIPoolingLayer : public Layer<Dtype> {
 public:
  explicit PSROIPoolingLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PSROIPooling"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype spatial_scale_;
  int output_dim_;
  int group_size_;

  int channels_;
  int height_;
  int width_;

  int pooled_height_;
  int pooled_width_;
  Blob<int> mapping_channel_;
};

}  // namespace caffe

#endif  // CAFFE_PSROI_POOLING_LAYER_HPP_
