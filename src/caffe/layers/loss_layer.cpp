#include <vector>

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void LossLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void LossLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "The data and label should have the same first dimension.";
  vector<int_tp> loss_shape(1, 1);  // Loss layers output a scalar; 1 element.
  top[0]->Reshape(loss_shape);
}

INSTANTIATE_CLASS_3T_GUARDED(LossLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(LossLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(LossLayer, (double), (double), (double));

}  // namespace caffe
