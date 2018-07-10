#include <algorithm>
#include <vector>

#include "caffe/layers/hinge_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void HingeLossLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int_tp num = bottom[0]->shape(0);
  int_tp count = bottom[0]->count();
  int_tp dim = count / num;

  caffe_copy(count, bottom_data, bottom_diff);
  for (int_tp i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int_tp>(label[i])] *= -1;
  }
  for (int_tp i = 0; i < num; ++i) {
    for (int_tp j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] = fmax(
        Dtype(0), Dtype(Dtype(1) + bottom_diff[i * dim + j]));
    }
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  switch (this->layer_param_.hinge_loss_param().norm()) {
  case HingeLossParameter_Norm_L1:
    loss[0] = caffe_asum(count, bottom_diff) / num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_dot(count, bottom_diff, bottom_diff) / num;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void HingeLossLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int_tp num = bottom[0]->shape(0);
    int_tp count = bottom[0]->count();
    int_tp dim = count / num;

    for (int_tp i = 0; i < num; ++i) {
      bottom_diff[i * dim + static_cast<int_tp>(label[i])] *= -1;
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    switch (this->layer_param_.hinge_loss_param().norm()) {
    case HingeLossParameter_Norm_L1:
      caffe_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, Dtype(loss_weight / num), bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      caffe_scal(count, Dtype(loss_weight * 2 / num), bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  }
}

INSTANTIATE_CLASS_3T_GUARDED(HingeLossLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(HingeLossLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(HingeLossLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(HingeLoss);
REGISTER_LAYER_CLASS_INST(HingeLoss, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(HingeLoss, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(HingeLoss, (double), (double), (double));

}  // namespace caffe
