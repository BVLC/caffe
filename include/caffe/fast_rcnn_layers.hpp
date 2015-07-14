// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#ifndef CAFFE_FAST_RCNN_LAYERS_HPP_
#define CAFFE_FAST_RCNN_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/* ROIPoolingLayer - Region of Interest Pooling Layer
*/
template <typename Dtype>
class ROIPoolingLayer : public Layer<Dtype> {
 public:
  explicit ROIPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ROIPooling"; }

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

  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  Dtype spatial_scale_;
  Blob<int> max_idx_;
};

template <typename Dtype>
class SmoothL1LossLayer : public LossLayer<Dtype> {
 public:
  explicit SmoothL1LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SmoothL1Loss"; }

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  /**
   * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> errors_;
  bool has_weights_;
};

}  // namespace caffe

#endif  // CAFFE_FAST_RCNN_LAYERS_HPP_
