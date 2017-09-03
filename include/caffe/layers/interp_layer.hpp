#ifndef CAFFE_INTERP_LAYER_HPP_
#define CAFFE_INTERP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Changes the spatial resolution by bi-linear interpolation.
 *        The target size is specified in terms of pixels. 
 *        The start and end pixels of the input are mapped to the start
 *        and end pixels of the output.
 */
template <typename Dtype>
class InterpLayer : public Layer<Dtype> {
 public:
  explicit InterpLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Interp"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  int num_, channels_;
  int height_in_, width_in_;
  int height_out_, width_out_;
  int pad_beg_, pad_end_;
  int height_in_eff_, width_in_eff_;
};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_
