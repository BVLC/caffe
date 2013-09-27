// Copyright 2013 Yangqing Jia

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include <algorithm>
#include <cmath>

using std::max;

namespace caffe {

template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "Loss Layer takes no as output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
};


template <typename Dtype>
Dtype MultinomialLogisticLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* bottom_label = (*bottom)[1]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
  memset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count());
  Dtype loss = 0;
  const Dtype kLOG_THRESHOLD = 1e-8;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    Dtype prob = max(bottom_data[i * dim + label], kLOG_THRESHOLD);
    loss -= log(prob);
    bottom_diff[i * dim + label] = - 1. / prob / num;
  }
  return loss / num;
}

// TODO: implement the GPU version

INSTANTIATE_CLASS(MultinomialLogisticLossLayer);


}  // namespace caffe
