// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Neuron Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Neuron Layer takes a single blob as output.";
  // NeuronLayer allows in-place computations. If the computation is not
  // in-place, we will need to initialize the top blob.
  if ((*top)[0] != bottom[0]) {
    (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
  }
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
