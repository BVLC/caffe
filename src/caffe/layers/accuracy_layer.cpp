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

using std::max;

template <typename Dtype>
void AccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  top_k_ = this->layer_param_.accuracy_param().top_k();
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
Dtype AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  for (int i = 0; i < num; ++i) {
    // Top-k accuracy
    std::fill_n(maxval.begin(), top_k_, -FLT_MAX);
    std::fill_n(max_id.begin(), top_k_, 0);
    for (int j = 0, k; j < dim; ++j) {
      // insert into (reverse-)sorted top-k array
      Dtype val = bottom_data[i * dim + j];
      for (k = top_k_; k > 0 && maxval[k-1] < val; k--) {
        maxval[k] = maxval[k-1];
        max_id[k] = max_id[k-1];
      }
      maxval[k] = val;
      max_id[k] = j;
    }
    // check if true label is in top k predictions
    for (int k = 0; k < top_k_; k++)
      if (max_id[k] == static_cast<int>(bottom_label[i])) {
        ++accuracy;
        break;
      }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;

  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(AccuracyLayer);

}  // namespace caffe
