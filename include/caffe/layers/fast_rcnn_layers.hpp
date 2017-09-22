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
//#include "caffe/loss_layers.hpp"
#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/gen_anchors.hpp"

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

/* SimplerNMSLayer - N Mini-batch Sampling Layer
*/
template <typename Dtype>
class SimplerNMSLayer : public Layer<Dtype> {
public:
    SimplerNMSLayer(const LayerParameter& param) :Layer<Dtype>(param),
        max_proposals_(500),
        prob_threshold_(0.5f),
        iou_threshold_(0.7f),
        min_bbox_size_(16),
        feat_stride_(16),
        pre_nms_topN_(6000),
        post_nms_topN_(300) {
    };

    ~SimplerNMSLayer() {
    }

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        top[0]->Reshape(std::vector<int>{ max_proposals_, 5 });
    }

    virtual inline const char* type() const { return "SimplerNMS"; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
    int max_proposals_;
    float prob_threshold_;
    // TODO: add to proto
    float iou_threshold_;
    int min_bbox_size_;
    int feat_stride_;
    int pre_nms_topN_;
    int post_nms_topN_;

    // relative to center point,
    // currently, it is always float, just do a quick fix
    Blob<float> anchors_blob_;

    //TODO: clamp is part of std as of c++17...
    constexpr static inline const Dtype clamp_v(const Dtype v, const Dtype v_min, const Dtype v_max)
    {
        return std::max(v_min, std::min(v, v_max));
    }
    struct simpler_nms_roi_t
    {
        Dtype x0, y0, x1, y1;

        Dtype area() const { return std::max<Dtype>(0, y1 - y0 + 1) * std::max<Dtype>(0, x1 - x0 + 1); }
        simpler_nms_roi_t intersect (simpler_nms_roi_t other) const
        {
            return
            {
                std::max(x0, other.x0),
                std::max(y0, other.y0),
                std::min(x1, other.x1),
                std::min(y1, other.y1)
            };
        }
        simpler_nms_roi_t clamp (simpler_nms_roi_t other) const
        {
            return
            {
                clamp_v(x0, other.x0, other.x1),
                clamp_v(y0, other.y0, other.y1),
                clamp_v(x1, other.x0, other.x1),
                clamp_v(y1, other.y0, other.y1)
            };
        }
    };

    struct simpler_nms_delta_t { Dtype shift_x, shift_y, log_w, log_h; };
    struct simpler_nms_proposal_t { simpler_nms_roi_t roi; Dtype confidence; size_t ord; };

    static std::vector<simpler_nms_roi_t> simpler_nms_perform_nms(
            const std::vector<simpler_nms_proposal_t>& proposals,
            float iou_threshold,
            size_t top_n);

    static void sort_and_keep_at_most_top_n(
            std::vector<simpler_nms_proposal_t>& proposals,
            size_t top_n);

    static simpler_nms_roi_t simpler_nms_gen_bbox(
            const anchor& box,
            const simpler_nms_delta_t& delta,
            int anchor_shift_x,
            int anchor_shift_y);
};

}  // namespace caffe

#endif  // CAFFE_FAST_RCNN_LAYERS_HPP_
