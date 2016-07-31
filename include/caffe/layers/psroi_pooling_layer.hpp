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
