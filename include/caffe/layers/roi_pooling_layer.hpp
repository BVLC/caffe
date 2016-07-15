#ifndef CAFFE_ROI_POOLING_LAYER_HPP_
#define CAFFE_ROI_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/** 
 * @brief Perform max pooling on regions of interest specified by input, takes
 *        as input N feature maps and a list of R regions of interest.
 *
 *   ROIPoolingLayer takes 2 inputs and produces 1 output. bottom[0] is
 *   [N x C x H x W] feature maps on which pooling is performed. bottom[1] is
 *   [R x 5] containing a list R ROI tuples with batch index and coordinates of
 *   regions of interest. Each row in bottom[1] is a ROI tuple in format
 *   [batch_index x1 y1 x2 y2], where batch_index corresponds to the index of
 *   instance in the first input and x1 y1 x2 y2 are 0-indexed coordinates
 *   of ROI rectangle (including its boundaries).
 *
 *   For each of the R ROIs, max-pooling is performed over pooled_h x pooled_w
 *   output bins (specified in roi_pooling_param). The pooling bin sizes are
 *   adaptively set such that they tile ROI rectangle in the indexed feature
 *   map. The pooling region of vertical bin ph in [0, pooled_h) is computed as
 *
 *    start_ph (included) = y1 + floor(ph * (y2 - y1 + 1) / pooled_h)
 *    end_ph (excluded)   = y1 + ceil((ph + 1) * (y2 - y1 + 1) / pooled_h)
 *
 *   and similar horizontal bins.
 *
 * @param param provides ROIPoolingParameter roi_pooling_param,
 *        with ROIPoolingLayer options:
 *  - pooled_h. The pooled output height.
 *  - pooled_w. The pooled output width
 *  - spatial_scale. Multiplicative spatial scale factor to translate ROI
 *  coordinates from their input scale to the scale used when pooling.
 *
 * Fast R-CNN
 * Written by Ross Girshick
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

}  // namespace caffe

#endif  // CAFFE_ROI_POOLING_LAYER_HPP_
