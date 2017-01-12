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

#ifndef CAFFE_PRIORBOX_LAYER_HPP_
#define CAFFE_PRIORBOX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Generate the prior boxes of designated sizes and aspect ratios across
 *        all dimensions @f$ (H \times W) @f$.
 *
 * Intended for use with MultiBox detection method to generate prior (template).
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class PriorBoxLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides PriorBoxParameter prior_box_param,
   *     with PriorBoxLayer options:
   *   - min_size (\b minimum box size in pixels. can be multiple. required!).
   *   - max_size (\b maximum box size in pixels. can be ignored or same as the
   *   # of min_size.).
   *   - aspect_ratio (\b optional aspect ratios of the boxes. can be multiple).
   *   - flip (\b optional bool, default true).
   *     if set, flip the aspect ratio.
   */
  explicit PriorBoxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PriorBox"; }
  virtual inline int ExactBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @brief Generates prior boxes for a layer with specified parameters.
   *
   * @param bottom input Blob vector (at least 2)
   *   -# @f$ (N \times C \times H_i \times W_i) @f$
   *      the input layer @f$ x_i @f$
   *   -# @f$ (N \times C \times H_0 \times W_0) @f$
   *      the data layer @f$ x_0 @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times 2 \times K*4) @f$ where @f$ K @f$ is the prior numbers
   *   By default, a box of aspect ratio 1 and min_size and a box of aspect
   *   ratio 1 and sqrt(min_size * max_size) are created.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
  }

  vector<float> min_sizes_;
  vector<float> max_sizes_;
  vector<float> aspect_ratios_;
  bool flip_;
  int num_priors_;
  bool clip_;
  vector<float> variance_;

  int img_w_;
  int img_h_;
  float step_w_;
  float step_h_;

  float offset_;
};

}  // namespace caffe

#endif  // CAFFE_PRIORBOX_LAYER_HPP_
