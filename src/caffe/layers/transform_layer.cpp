// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/tranform_layers.hpp"

namespace caffe {

template <typename Dtype>
void TransformLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  // TransformLayer should maintain the same number of blobs
  CHECK_EQ(bottom.size(), top->size()) <<
  	"TransformLayers should have the same number of bottom and top blobs";
}

INSTANTIATE_CLASS(TransformLayer);

}  // namespace caffe
