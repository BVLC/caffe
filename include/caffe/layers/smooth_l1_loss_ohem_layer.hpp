#ifndef CAFFE_SMOOTH_L1_LOSS_OHEM_LAYER_HPP_
#define CAFFE_SMOOTH_L1_LOSS_OHEM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief SmoothL1LossOHEMLayer
 *
 * R-FCN
 * Written by Yi Li
 */
  template <typename Dtype>
  class SmoothL1LossOHEMLayer : public LossLayer<Dtype> {
   public:
    explicit SmoothL1LossOHEMLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "SmoothL1LossOHEM"; }

    virtual inline int ExactNumBottomBlobs() const { return -1; }
    virtual inline int MinBottomBlobs() const { return 2; }
    virtual inline int MaxBottomBlobs() const { return 3; }
    virtual inline int ExactNumTopBlobs() const { return -1; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 2; }

    /**
    * Unlike most loss layers, in the SmoothL1LossOHEMLayer we can backpropagate
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

    /// Read the normalization mode parameter and compute the normalizer based
    /// on the blob size.
    virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode,
      Dtype pre_fixed_normalizer);

    Blob<Dtype> diff_;
    Blob<Dtype> errors_;
    bool has_weights_;

    int outer_num_, inner_num_;

    /// How to normalize the output loss.
    LossParameter_NormalizationMode normalization_;
  };

}  // namespace caffe

#endif  // CAFFE_SMOOTH_L1_LOSS_OHEM_LAYER_HPP_
