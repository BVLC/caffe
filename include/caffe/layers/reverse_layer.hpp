#ifndef REVERSE_LAYER_HPP
#define REVERSE_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/*
 * @brief Reverses the data of the input Blob into the output blob.
 *
 * Note: This is a useful layer if you want to reverse the time of
 * a recurrent layer.
 */

template <typename Dtype>
class ReverseLayer : public NeuronLayer<Dtype> {
 public:
  explicit ReverseLayer(const LayerParameter& param);

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Reverse"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int axis_;
};

}  // namespace caffe

#endif  // REVERSE_LAYER_HPP
