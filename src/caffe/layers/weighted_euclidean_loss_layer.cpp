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

  Dtype wdot(0.0);
  for (int i = 0; i < count; ++i) {
    wdot += bottom[2]->cpu_data()[i] *
      diff_.cpu_data()[i] * diff_.cpu_data()[i];
  }

  Dtype loss = wdot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) {
    LOG(FATAL) << this->type() << 
     "Weighted Euclidean loss layer cannot backpropagate to certainty inputs.";
  }
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
      for (int j = 0; j < bottom[i]->count(); ++j) {
        bottom[i]->mutable_cpu_diff()[j] *= bottom[2]->cpu_data()[j];
      }
    }
  }
}

INSTANTIATE_CLASS(WeightedEuclideanLossLayer);
REGISTER_LAYER_CLASS(WeightedEuclideanLoss);

}  // namespace caffe
