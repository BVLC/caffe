#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void EuclideanLossLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  LossLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template<typename Dtype, typename MItype, typename MOtype>
void EuclideanLossLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  int_tp count = bottom[0]->count();
  caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
            diff_.mutable_cpu_data());
  // Scale the error element-wise
  if (bottom.size() == 3) {
    caffe_mul<Dtype>(count, diff_.mutable_cpu_data(), bottom[2]->cpu_data(),
                     diff_.mutable_cpu_data());
  }
  Dtype dot = caffe_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / static_cast<Dtype>(bottom[0]->count(0)) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype, typename MItype, typename MOtype>
void EuclideanLossLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  for (int_tp i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0]
          / static_cast<Dtype>(bottom[0]->count(0));
      caffe_axpby(bottom[i]->count(),     // count
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

INSTANTIATE_CLASS_3T_GUARDED(EuclideanLossLayer,
                             (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(EuclideanLossLayer,
                             (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(EuclideanLossLayer,
                             (double), (double), (double));

REGISTER_LAYER_CLASS(EuclideanLoss);
REGISTER_LAYER_CLASS_INST(EuclideanLoss,
                          (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(EuclideanLoss,
                          (float), (float), (float));
REGISTER_LAYER_CLASS_INST(EuclideanLoss,
                          (double), (double), (double));

}  // namespace caffe
