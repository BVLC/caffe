#include <vector>

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void NeuronLayer<Dtype, MItype, MOtype>::Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

INSTANTIATE_CLASS_3T(NeuronLayer);

}  // namespace caffe
