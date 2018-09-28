#ifndef CAFFE_EVAL_DETECTION_LAYER_HPP_
#define CAFFE_EVAL_DETECTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class EvalDetectionLayer : public Layer<Dtype> {
 public:
  explicit EvalDetectionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EvalDetection"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  int side_;
  int num_class_;
  int num_object_;
  float threshold_;
  //bool sqrt_;
  //bool constriant_;
  int score_type_;
  float nms_;
  vector<Dtype> biases_;
};

}  // namespace caffe

#endif  // CAFFE_EVAL_DETECTION_LAYER_HPP_
