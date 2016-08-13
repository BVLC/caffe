#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sum_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SumLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  Dtype sum;
  caffe_gpu_dot(count, bottom[0]->gpu_data(), ones_.gpu_data(), &sum);
//   LOG(INFO) << sum;
  Dtype loss = sum / bottom[0]->num();
//   LOG(INFO) << loss;
  (top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SumLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype alpha = top[0]->cpu_diff()[0] / (bottom)[0]->num();
    caffe_gpu_axpby(
        (bottom)[0]->count(),              // count
        alpha,                              // alpha
        ones_.gpu_data(),                   // a
        Dtype(0),                           // beta
        (bottom)[0]->mutable_gpu_diff());  // b
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SumLossLayer);

}  // namespace caffe
