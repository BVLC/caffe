#ifndef CAFFE_BOX_ANNOTATOR_OHEM_LAYER_HPP_
#define CAFFE_BOX_ANNOTATOR_OHEM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

 /**
 * @brief BoxAnnotatorOHEMLayer: Annotate box labels for Online Hard Example Mining (OHEM) training
 * R-FCN
 * Written by Yi Li
 */
  template <typename Dtype>
  class BoxAnnotatorOHEMLayer :public Layer<Dtype>{
   public:
    explicit BoxAnnotatorOHEMLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "BoxAnnotatorOHEM"; }

    virtual inline int ExactNumBottomBlobs() const { return 4; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int num_;
    int height_;
    int width_;
    int spatial_dim_;
    int bbox_channels_;

    int roi_per_img_;
    int ignore_label_;
  };

}  // namespace caffe

#endif  // CAFFE_BOX_ANNOTATOR_OHEM_LAYER_HPP_
