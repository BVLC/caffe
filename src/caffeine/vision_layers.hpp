#ifndef CAFFEINE_VISION_LAYERS_HPP_
#define CAFFEINE_VISION_LAYERS_HPP_

#include "caffeine/layer.hpp"

namespace caffeine {

template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
 public:
  virtual void SetUp(vector<const Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
};

template <typename Dtype>
class ReLULayer : public NeuronLayer<Dtype> {
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  virtual void Predict_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top);
  virtual void Predict_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
};

}  // namespace caffeine

#endif  // CAFFEINE_VISION_LAYERS_HPP_

