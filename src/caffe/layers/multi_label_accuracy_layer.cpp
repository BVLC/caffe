// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "MultiLabelAccuracy Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 1) << "MultiLabelAccuracy Layer takes 1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number of instances";
  CHECK_EQ(bottom[1]->channels(), bottom[1]->channels())
    << "The data and label should have the same number of channels";
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 2, 1, 1);
}

template <typename Dtype>
Dtype MultiLabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logloss = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();

  caffe_copy(count, bottom_data, bottom_diff);
  caffe_cpu_sign(count, bottom_diff, bottom_diff);
  for (int i = 0; i < num; ++i) {
    // Accuracy
    for (int j = 0; j < dim; ++j) {
      int ind = i * dim + j;
      int label = static_cast<int>(bottom_label[ind]);
      if (label != 0) {
        accuracy += (bottom_diff[ind] == label);
        logloss -= bottom_data[ind] * ((label >= 0) - (bottom_data[ind] >= 0)) -
          log(1 + exp(bottom_data[ind] - 2 * bottom_data[ind] *
          (bottom_data[ind] >= 0)));
      }
    }
  }
  LOG(INFO) << "Accuracy: " << accuracy;
  LOG(INFO) << "Logloss: " << logloss;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / count;
  (*top)[0]->mutable_cpu_data()[1] = logloss / count;
  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);

}  // namespace caffe
