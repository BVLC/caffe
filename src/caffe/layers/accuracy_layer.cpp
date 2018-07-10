#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void AccuracyLayer<Dtype, MItype, MOtype>::LayerSetUp(
  const vector<Blob<MItype>*>& bottom,
  const vector<Blob<MOtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }

  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void AccuracyLayer<Dtype, MItype, MOtype>::Reshape(
  const vector<Blob<MItype>*>& bottom,
  const vector<Blob<MOtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (n, c, H, W), "
      << "label count (number of labels) must be n*H*W, "
      << "with integer values in {0, 1, ..., c-1}.";
  vector<int_tp> top_shape(1, 1);  // Accuracy is a scalar; 1 element.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int_tp> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void AccuracyLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int_tp dim = bottom[0]->count() / outer_num_;
  const int_tp num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int_tp> max_id(top_k_+1);
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int_tp count = 0;
  for (int_tp i = 0; i < outer_num_; ++i) {
    for (int_tp j = 0; j < inner_num_; ++j) {
      const int_tp label_value =
          static_cast<int_tp>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
      const Dtype prob_of_true_class = bottom_data[i * dim
                                                   + label_value * inner_num_
                                                   + j];
      int num_better_predictions = -1;  // true_class also counts as "better"
      // Top-k accuracy
      for (int k = 0; k < num_labels && num_better_predictions < top_k_; ++k) {
        num_better_predictions +=
          (bottom_data[i * dim + k * inner_num_ + j] >= prob_of_true_class);
      }
      // check if there are less than top_k_ predictions
      if (num_better_predictions < top_k_) {
        ++accuracy;
        if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
      }
      ++count;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = (count == 0) ? Dtype(0) :
      (accuracy / static_cast<Dtype>(count));
  if (top.size() > 1) {
    for (int_tp i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == Dtype(0) ? Dtype(0)
          : (top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i]);
    }
  }
  // Accuracy layer should not be used as a loss function.
}

#ifdef CPU_ONLY
STUB_GPU(AccuracyLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(AccuracyLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(AccuracyLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(AccuracyLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Accuracy);
REGISTER_LAYER_CLASS_INST(Accuracy, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Accuracy, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Accuracy, (double), (double), (double));

}  // namespace caffe
