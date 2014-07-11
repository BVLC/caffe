#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // NeuronLayer allows in-place computations. If the computation is not
  // in-place, we will need to initialize the top blob.
  if ((*top)[0] != bottom[0]) {
    (*top)[0]->ReshapeLike(*bottom[0]);
  }
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
