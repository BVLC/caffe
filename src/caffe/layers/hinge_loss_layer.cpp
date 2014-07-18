// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
Dtype HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
  }
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] = std::max(
        Dtype(0), 1 + bottom_diff[i * dim + j]);
    }
  }
  switch (this->layer_param_.hinge_loss_param().norm()) {
  case HingeLossParameter_Norm_L1:
    return caffe_cpu_asum(count, bottom_diff) / num;
  case HingeLossParameter_Norm_L2:
    return caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template <typename Dtype>
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = (*bottom)[0]->num();
    int count = (*bottom)[0]->count();
    int dim = count / num;

    for (int i = 0; i < num; ++i) {
      bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
    }

    switch (this->layer_param_.hinge_loss_param().norm()) {
    case HingeLossParameter_Norm_L1:
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, Dtype(1. / num), bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      caffe_scal(count, Dtype(2. / num), bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  }
}

INSTANTIATE_CLASS(HingeLossLayer);

}  // namespace caffe
