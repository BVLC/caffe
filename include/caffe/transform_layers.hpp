// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_TRANSFORM_LAYERS_HPP_
#define CAFFE_TRANSFORM_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "pthread.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/* TransformationLayer

  An interface for layers that take a vector of blobs as input (x),
  and produce a vector of transformed blobs as output (y=f(x)).
  For now meant to be used within the data layer for preprocessing
  For example Crop_Mirror, Center_Scale, ...

*/
template <typename Dtype>
class TransformationLayer : public Layer<Dtype> {
 public:
  explicit TransformationLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_NONE;
  }
  virtual inline int MinNumBottomBlobs() const { return 1; }
  virtual inline int MinNumTopBlobs() const { return 1; }
};

/* Crop_Mirror

  Allows to crop portions of the blobs and flip horizontally at
  the same time. 

*/
template <typename Dtype>
class CropMirrorLayer : public TransformationLayer<Dtype> {
 public:
  explicit CropMirrorLayer(const LayerParameter& param)
      : TransformationLayer<Dtype>(param) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_CROP_MIRROR;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
  int crop_size_;
  bool mirror_;
};

/* Center_Scale

  Allows to center values by removing a predifined mean value/s or mean_blob.
  Also allows to scale the values at the same time.

*/
template <typename Dtype>
class CenterScaleLayer : public TransformationLayer<Dtype> {
 public:
  explicit CenterScaleLayer(const LayerParameter& param)
      : TransformationLayer<Dtype>(param) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_CENTER_SCALE;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;

  Blob<Dtype> data_mean_;
  bool has_mean_file_;
  bool has_scale_;
  Dtype scale_;
  std::vector<Dtype> mean_values_;
  std::vector<Dtype> scale_values_;

};

/* Color Jittering 

  Creates random perturbation of the color space

*/
template <typename Dtype>
class ColorJitteringLayer : public TransformationLayer<Dtype> {
 public:
  explicit ColorJitteringLayer(const LayerParameter& param)
      : TransformationLayer<Dtype>(param) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_COLOR_JITTERING;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;

};

}  // namespace caffe

#endif  // CAFFE_TRANSFORM_LAYERS_HPP_
