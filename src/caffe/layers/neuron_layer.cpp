#include <vector>

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void NeuronLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

INSTANTIATE_CLASS_3T_GUARDED(NeuronLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(NeuronLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(NeuronLayer, (double), (double), (double));

}  // namespace caffe
