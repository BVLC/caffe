#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/confusion_matrix_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConfusionMatrixLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ConfusionMatrixLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  label_axis_ = bottom[0]->CanonicalAxisIndex(
                  this->layer_param_.confusion_matrix_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  num_classes_ = bottom[0]->shape(label_axis_);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of channels in bottom[0]";
  vector<int> confusion_dim(2, num_classes_);
  top[0]->Reshape(confusion_dim);
  confusion_matrix_.Reshape(confusion_dim);
}

template <typename Dtype>
void ConfusionMatrixLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;

  caffe_set(confusion_matrix_.count(), Dtype(0),
            confusion_matrix_.mutable_cpu_data());
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());

  Dtype max_val = -1.;
  int max_id = 0;
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_classes_);

      max_id = 0;
      max_val = -1.;
      for (int k = 0; k < num_classes_; ++k) {
        if (bottom_data[i * dim + k * inner_num_ + j] > max_val) {
          max_val = bottom_data[i * dim + k * inner_num_ + j];
          max_id = k;
        }
      }
      int predicted_class = max_id;
      DCHECK_GE(predicted_class, 0);
      DCHECK_LT(predicted_class, num_classes_);
      confusion_matrix_.mutable_cpu_data()[label_value * num_classes_
                                           + predicted_class] += 1;
      ++count;
    }
  }

  int num_image_per_class = 0;
  for (int i = 0; i < num_classes_; ++i) {
    num_image_per_class = 0;
    int base= i * num_classes_;
    for (int j = 0; j < num_classes_; ++j) {
      num_image_per_class = num_image_per_class
                            + confusion_matrix_.cpu_data()[base +j];
    }
    for (int j = 0; j < num_classes_; ++j) {
      Dtype value = confusion_matrix_.cpu_data()[base + j]
                    / ((Dtype) num_image_per_class);
      top[0]->mutable_cpu_data()[base + j] = value;
    }
  }
}

INSTANTIATE_CLASS(ConfusionMatrixLayer);
REGISTER_LAYER_CLASS(ConfusionMatrix);

}  // namespace caffe
