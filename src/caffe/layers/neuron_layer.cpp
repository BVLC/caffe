#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  // NeuronLayer allows in-place computations. If the computation is not
  // in-place, we will need to initialize the top blob.
  if ((*top)[0] != bottom[0]) {
    (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
  }
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
