#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  // Scale each subtraction. Labels are in bottom[1].
  caffe_gpu_scal(count, scalar_, bottom[1]->mutable_gpu_data());
  caffe_gpu_mul(
      count,
      bottom[1]->gpu_data(),
      diff_.gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / num / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
  std::cout << "GPU Loss: " << loss << endl;
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / (*bottom)[i]->num();
      caffe_gpu_axpby(
          (*bottom)[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          (*bottom)[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_CLASS(WeightedEuclideanLossLayer);

}  // namespace caffe
