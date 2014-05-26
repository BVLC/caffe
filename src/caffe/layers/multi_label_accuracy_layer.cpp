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
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2)
    << "MultiLabelAccuracy Layer takes two blobs as input.";
  CHECK_LE(top->size(), 1)
    << "MultiLabelAccuracy Layer takes 0/1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number of instances";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
    << "The data and label should have the same number of channels";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data and label should have the same height";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data and label should have the same width";
  if (top->size() == 1) {
    // If top is used then it will contain:
    // top[0] = Sensitivity (TP/P), top[1] = Specificity (TN/N),
    // top[2] = Likehood ratio positive (Sen/(1-Spe)), top[2] = Loss
    (*top)[0]->Reshape(1, 4, 1, 1);
  }
}

template <typename Dtype>
Dtype MultiLabelAccuracyLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Dtype true_positive = 0;
  Dtype true_negative = 0;
  Dtype positive_weight_ =
    this->layer_param_.multi_label_accuracy_param().positive_weight();
  Dtype negative_weight_ =
    this->layer_param_.multi_label_accuracy_param().negative_weight();
  Dtype total_loss = 0;
  int count_pos = 0;
  int count_neg = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();

  for (int ind = 0; ind < count; ++ind) {
    // Accuracy
    int label = static_cast<int>(bottom_label[ind]);
    Dtype loss = 0;
    if (label != 0) {
    // Update the loss only if label is not 0
      loss = bottom_data[ind] * ((label > 0) - (bottom_data[ind] >= 0))
        - log(1 + exp(bottom_data[ind] - 2 * bottom_data[ind] *
        (bottom_data[ind] >= 0)));
    }
    if (label > 0) {
    // Update Positive accuracy and count
      true_positive += (bottom_data[ind] >= 0);
      count_pos++;
      loss *= positive_weight_;
    }
    if (label < 0) {
    // Update Negative accuracy and count
      true_negative += (bottom_data[ind] < 0);
      count_neg++;
      loss *= negative_weight_;
    }
    total_loss -= loss;
  }
  // LOG(INFO) << "Sensitivity: " << (true_positive / count_pos);
  // LOG(INFO) << "Specificity: " << (true_negative / count_neg);
  // LOG(INFO) << "Likehood ratio positive: " <<
  //  (true_positive / count_pos) / (1.0 - true_negative / count_neg);
  // LOG(INFO) << "Loss: " << total_loss;
  if ((top->size() == 1)) {
    (*top)[0]->mutable_cpu_data()[0] = true_positive / count_pos;
    (*top)[0]->mutable_cpu_data()[1] = true_negative / count_neg;
    (*top)[0]->mutable_cpu_data()[2] =
      (true_positive / count_pos) / (1.0 - true_negative / count_neg);
    (*top)[0]->mutable_cpu_data()[3] = total_loss / num;
  }
  // MultiLabelAccuracy can be used as a loss function.
  return Dtype(total_loss / num);
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* bottom_label = (*bottom)[1]->cpu_data();
  const int count = (*bottom)[0]->count();
  const int num = (*bottom)[0]->num();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  for (int i = 0; i < count; ++i) {
    if (bottom_label[i] != 0) {
      bottom_diff[i] = sigmoid(bottom_data[i]) - (bottom_label[i] > 0);
    } else {
      bottom_diff[i] = 0;
    }
  }
  // Scale down gradient
  caffe_scal(count, Dtype(1) / num, bottom_diff);
}


INSTANTIATE_CLASS(MultiLabelAccuracyLayer);

}  // namespace caffe
