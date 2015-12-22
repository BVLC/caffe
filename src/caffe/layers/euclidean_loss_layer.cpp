#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  int_tp count = bottom[0]->count();
  caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
            diff_.mutable_cpu_data());
  // Scale the error element-wise
  if (bottom.size() == 3) {
    caffe_mul<Dtype>(count, diff_.mutable_cpu_data(), bottom[2]->gpu_data(),
                     diff_.mutable_cpu_data());
  }
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / static_cast<Dtype>(bottom[0]->count(0)) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int_tp i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0]
          / static_cast<Dtype>(bottom[0]->count(0));
      caffe_cpu_axpby(bottom[i]->count(),     // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());     // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
