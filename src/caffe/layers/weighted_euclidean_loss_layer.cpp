#include <vector>

#include "caffe/layers/weighted_euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  sqrt_weight_.ReshapeLike(*bottom[0]);
  sqrt_weight_diff_.ReshapeLike(*bottom[0]);
  temp_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_sqrt(count,
             bottom[2]->cpu_data(),
             sqrt_weight_.mutable_cpu_data());
  caffe_mul(count,
            sqrt_weight_.cpu_data(),
            diff_.cpu_data(),
            sqrt_weight_diff_.mutable_cpu_data());
  Dtype wdot = caffe_cpu_dot(count,
                             sqrt_weight_diff_.cpu_data(),
                             sqrt_weight_diff_.cpu_data());

  Dtype loss = wdot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),   // count
          alpha,                // alpha
          diff_.cpu_data(),     // a
          Dtype(0),             // beta
          temp_.mutable_cpu_data());               // b

      caffe_mul(bottom[i]->count(),
                temp_.cpu_data(),
                bottom[2]->cpu_data(),
                bottom[i]->mutable_cpu_diff());
    }
  }
  // Propagate to weight layer
  if (propagate_down[2]) {
    caffe_mul(
        bottom[0]->count(),
        diff_.cpu_data(),
        diff_.cpu_data(),
        bottom[2]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(WeightedEuclideanLossLayer);
REGISTER_LAYER_CLASS(WeightedEuclideanLoss);

}  // namespace caffe
