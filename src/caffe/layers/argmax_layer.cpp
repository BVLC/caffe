// Copyright 2014 BVLC and contributors.

#include <vector>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
void ArgMaxLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  out_max_val_ = this->layer_param_.argmax_param().out_max_val();
  if (out_max_val_) {
    // Produces max_ind and max_val
    (*top)[0]->Reshape(bottom[0]->num(), 2, 1, 1);
  } else {
    // Produces only max_ind
    (*top)[0]->Reshape(bottom[0]->num(), 1, 1, 1);
  }
}

template <typename Dtype>
Dtype ArgMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    Dtype max_val = -FLT_MAX;
    int max_ind = 0;
    for (int j = 0; j < dim; ++j) {
      if (bottom_data[i * dim + j] > max_val) {
        max_val = bottom_data[i * dim + j];
        max_ind = j;
      }
    }
    if (out_max_val_) {
      top_data[i * 2] = max_ind;
      top_data[i * 2 + 1] = max_val;
    } else {
      top_data[i] = max_ind;
    }
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(ArgMaxLayer);


}  // namespace caffe
