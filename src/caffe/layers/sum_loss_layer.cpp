#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sum_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SumLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  
  ones_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  caffe_set(ones_.count(), Dtype(1), ones_.mutable_cpu_data());
}

template <typename Dtype>
void SumLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  Dtype sum = caffe_cpu_dot(count, bottom[0]->cpu_data(), ones_.cpu_data());
  Dtype loss = sum / bottom[0]->num();
  (top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SumLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype alpha = top[0]->cpu_diff()[0] / (bottom)[0]->num();
    caffe_cpu_axpby(
        (bottom)[0]->count(),              // count
        alpha,                              // alpha
        ones_.cpu_data(),                   // a
        Dtype(0),                           // beta
        (bottom)[0]->mutable_cpu_diff());  // b
  }
}

#ifdef CPU_ONLY
STUB_GPU(SumLossLayer);
#endif

INSTANTIATE_CLASS(SumLossLayer);
REGISTER_LAYER_CLASS(SumLoss);

}  // namespace caffe
