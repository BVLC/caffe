#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/multinomial_logistic_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void MultinomialLogisticLossLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  LossLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template<typename Dtype, typename MItype, typename MOtype>
void MultinomialLogisticLossLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int_tp num = bottom[0]->shape(0);
  int_tp dim = bottom[0]->count() / bottom[0]->shape(0);
  Dtype loss = 0;
  for (int_tp i = 0; i < num; ++i) {
    int_tp label = static_cast<int_tp>(bottom_label[i]);
    Dtype prob = std::max(
        bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
    loss -= std::log(prob);
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template<typename Dtype, typename MItype, typename MOtype>
void MultinomialLogisticLossLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int_tp num = bottom[0]->shape(0);
    int_tp dim = bottom[0]->count() / bottom[0]->shape(0);
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const Dtype scale = - top[0]->cpu_diff()[0] / num;
    for (int_tp i = 0; i < num; ++i) {
      int_tp label = static_cast<int_tp>(bottom_label[i]);
      Dtype prob = std::max(
          bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
      bottom_diff[i * dim + label] = scale / prob;
    }
  }
}

INSTANTIATE_CLASS_3T_GUARDED(MultinomialLogisticLossLayer,
                     (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(MultinomialLogisticLossLayer,
                     (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(MultinomialLogisticLossLayer,
                     (double), (double), (double));

REGISTER_LAYER_CLASS(MultinomialLogisticLoss);
REGISTER_LAYER_CLASS_INST(MultinomialLogisticLoss,
                          (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(MultinomialLogisticLoss,
                          (float), (float), (float));
REGISTER_LAYER_CLASS_INST(MultinomialLogisticLoss,
                          (double), (double), (double));

}  // namespace caffe
