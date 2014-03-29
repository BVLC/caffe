// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
Dtype RBMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                   vector<Blob<Dtype>*>* top) {
  return Dtype(0);
}

template<typename Dtype>
void RBMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                   const bool propagate_down,
                                   vector<Blob<Dtype>*>* bottom) {
}

INSTANTIATE_CLASS(RBMLayer);

}  // namespace caffe
