#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();
  denominator_ = this->layer_param_.accuracy_param().denominator();
  CHECK_GE(denominator_, 0)
      << "Denominator must be positive; or 0, for the batch size.";

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_GE(bottom[0]->num_axes(), bottom[1]->num_axes());
  for (int i = 0; i < bottom[1]->num_axes(); ++i) {
    CHECK_LE(bottom[0]->shape(i), bottom[1]->shape(i))
        << "Dimension mismatch between predictions and label.";
  }
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[1]->count();
  int dim = bottom[0]->count() / num;
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  int count = 0;
  for (int i = 0; i < num; ++i) {
    // Top-k accuracy
    std::vector<std::pair<Dtype, int> > bottom_data_vector;
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector.push_back(
          std::make_pair(bottom_data[i * dim + j], j));
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
    // check if true label is in top k predictions
    const int label_value = static_cast<int>(bottom_label[i]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      continue;
    }
    ++count;
    for (int k = 0; k < top_k_; k++) {
      if (bottom_data_vector[k].second == label_value) {
        ++accuracy;
        break;
      }
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  const Dtype denominator = (denominator_ == 0) ? count : denominator_;
  top[0]->mutable_cpu_data()[0] = accuracy / denominator;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
