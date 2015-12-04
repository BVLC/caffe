#include <vector>

<<<<<<< HEAD
#include "caffe/layers/neuron_layer.hpp"
=======
#include "caffe/neuron_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge

namespace caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
