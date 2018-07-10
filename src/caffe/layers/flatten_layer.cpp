#include <vector>

#include "caffe/layers/flatten_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void FlattenLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  const int_tp start_axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.flatten_param().axis());
  const int_tp end_axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.flatten_param().end_axis());
  vector<int_tp> top_shape;
  for (int_tp i = 0; i < start_axis; ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  const int_tp flattened_dim = bottom[0]->count(start_axis, end_axis + 1);
  top_shape.push_back(flattened_dim);
  for (int_tp i = end_axis + 1; i < bottom[0]->num_axes(); ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count());
}

template<typename Dtype, typename MItype, typename MOtype>
void FlattenLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template<typename Dtype, typename MItype, typename MOtype>
void FlattenLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom) {
  bottom[0]->ShareDiff(*top[0]);
}

INSTANTIATE_CLASS_3T_GUARDED(FlattenLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(FlattenLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(FlattenLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Flatten);
REGISTER_LAYER_CLASS_INST(Flatten, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Flatten, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Flatten, (double), (double), (double));

}  // namespace caffe
