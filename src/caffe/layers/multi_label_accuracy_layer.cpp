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
  (*top)[0]->Reshape(1, 4, 1, 1);
}

template <typename Dtype>
Dtype MultiLabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy_pos = 0;
  Dtype accuracy_neg = 0;
  Dtype logloss = 0;
  int count_pos = 0;
  int count_neg = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();

  // caffe_copy(count, bottom_data, bottom_diff);
  // caffe_cpu_sign(count, bottom_diff, bottom_diff);
  for (int i = 0; i < num; ++i) {
    // Accuracy
    for (int j = 0; j < dim; ++j) {
      int ind = i * dim + j;
      int label = static_cast<int>(bottom_label[ind]);
      // if (label == 0) {
      //   //Ignore
      //   continue;
      // }
      if (label == 1) { 
      // Positive
        accuracy_pos += (bottom_data[ind] >= 0);
        count_pos++;
      }
      if (label == 0) {
      // Negative
        accuracy_neg += (bottom_data[ind] < 0);
        count_neg++;        
      }
      logloss -= bottom_data[ind] * (label - (bottom_data[ind] >= 0)) -
        log(1 + exp(bottom_data[ind] - 2 * bottom_data[ind] *
        (bottom_data[ind] >= 0)));
    }
  }
  LOG(INFO) << "Accuracy: " << accuracy_pos << " " << accuracy_neg;
  LOG(INFO) << "Logloss: " << logloss;
  (*top)[0]->mutable_cpu_data()[0] = accuracy_pos / count_pos;
  (*top)[0]->mutable_cpu_data()[1] = accuracy_neg / count_neg;
  (*top)[0]->mutable_cpu_data()[2] = (accuracy_pos / count_pos +
    accuracy_neg / count_neg) / 2;
  (*top)[0]->mutable_cpu_data()[3] = logloss / num;
  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);

}  // namespace caffe
