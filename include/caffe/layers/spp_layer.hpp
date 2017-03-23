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

#ifndef CAFFE_SPP_LAYER_HPP_
#define CAFFE_SPP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Does spatial pyramid pooling on the input image
 *        by taking the max, average, etc. within regions
 *        so that the result vector of different sized
 *        images are of the same size.
 */
template <typename Dtype>
class SPPLayer : public Layer<Dtype> {
 public:
  explicit SPPLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  ~SPPLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SPP"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // calculates the kernel and stride dimensions for the pooling layer,
  // returns a correctly configured LayerParameter for a PoolingLayer
  virtual LayerParameter GetPoolingParam(const int pyramid_level,
      const int bottom_h, const int bottom_w, const SPPParameter spp_param);

  int pyramid_height_;
  int bottom_h_, bottom_w_;
  int num_;
  int channels_;
  int kernel_h_, kernel_w_;
  int pad_h_, pad_w_;
  bool reshaped_first_time_;

  /// the internal Split layer that feeds the pooling layers
  shared_ptr<SplitLayer<Dtype> > split_layer_;
  /// top vector holder used in call to the underlying SplitLayer::Forward
  vector<Blob<Dtype>*> split_top_vec_;
  /// bottom vector holder used in call to the underlying PoolingLayer::Forward
  vector<shared_ptr<vector<Blob<Dtype>*> > > pooling_bottom_vecs_;
  /// the internal Pooling layers of different kernel sizes
  vector<shared_ptr<PoolingLayer<Dtype> > > pooling_layers_;
  /// top vector holders used in call to the underlying PoolingLayer::Forward
  vector<shared_ptr<vector<Blob<Dtype>*> > > pooling_top_vecs_;
  /// pooling_outputs stores the outputs of the PoolingLayers
  vector<shared_ptr<Blob<Dtype> > > pooling_outputs_;
  /// the internal Flatten layers that the Pooling layers feed into
  vector<shared_ptr<FlattenLayer<Dtype> > > flatten_layers_;
  /// top vector holders used in call to the underlying FlattenLayer::Forward
  vector<shared_ptr<vector<Blob<Dtype>*> > > flatten_top_vecs_;
  /// flatten_outputs stores the outputs of the FlattenLayers
  vector<shared_ptr<Blob<Dtype> > > flatten_outputs_;
  /// bottom vector holder used in call to the underlying ConcatLayer::Forward
  vector<Blob<Dtype>*> concat_bottom_vec_;
  /// the internal Concat layers that the Flatten layers feed into
  shared_ptr<ConcatLayer<Dtype> > concat_layer_;
};

}  // namespace caffe

#endif  // CAFFE_SPP_LAYER_HPP_
