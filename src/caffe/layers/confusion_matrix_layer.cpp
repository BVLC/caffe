#include "caffe/layers/confusion_matrix_layer.hpp"

/*
 * How to use the layer:
 * append the layer to the last full connection layer
 * in the train_val.prototxt,
 * like how you do with "Accuracy" layer.

  layer {
  name: "confusion_matrix"
  type: "ConfusionMatrix"
  bottom: "fc7"
  bottom: "label"
    include {
      phase: TEST
    }
  }

  * The confusion matrix would be like:
  Confusion Matrix                          | Accuracy
  169       10      5       3       3       | 88.95%    => recall of class i
  3         260     3       3       1       | 96.30%
  0         1       174     4       15      | 89.69%
  3         6       1       195     5       | 92.86%
  1         1       26      2       210     | 87.50%
  96.02%    93.53%  83.25%  94.20%  89.74%  | 1104
  /\
  ||
  precision of class i

 */

namespace caffe {
  template<typename Dtype>
  void ConfusionMatrixLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

    CHECK_EQ(bottom.size(), 2) << "Need two inputs\n";
    CHECK_EQ(top.size(), 0) << "Zero output\n";
    current_iter = 0;

    has_ignore_label_ =
      this->layer_param_.confusion_matrix_param().has_ignore_label();
    if (has_ignore_label_) {
      ignore_label_ = this->layer_param_.confusion_matrix_param().ignore_label();
    }

    label_axis_ =
      bottom[0]->CanonicalAxisIndex(
        this->layer_param_.confusion_matrix_param().axis());
    num_labels = bottom[0]->shape(label_axis_);

    test_iter = this->layer_param_.confusion_matrix_param().test_iter();

    std::vector<int> top_shape(1);
    top_shape[0] = num_labels * num_labels;

    nums_buffer_.Reshape(top_shape);
    caffe_set(num_labels * num_labels, Dtype(0), nums_buffer_.mutable_cpu_data());
    LOG(INFO) << nums_buffer_.shape_string() << "\n";
  }

  template<typename Dtype>
  void ConfusionMatrixLayer<Dtype>::Reshape(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
    label_axis_ =
      bottom[0]->CanonicalAxisIndex(
        this->layer_param_.confusion_matrix_param().axis());

    CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "Inputs must have the same number of elements.\n";

    outer_num_ = bottom[0]->count(0, label_axis_);
    inner_num_ = bottom[0]->count(label_axis_ + 1);
    CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";

    num_labels = bottom[0]->shape(label_axis_);
  }

  template<typename Dtype>
  void ConfusionMatrixLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                                const vector<Blob<Dtype> *> &top) {
    Dtype accuracy = 0;
    const Dtype *bottom_data = bottom[0]->cpu_data();
    const Dtype *bottom_label = bottom[1]->cpu_data();
    Dtype *buffer_data = nums_buffer_.mutable_cpu_data();

    const int dim = bottom[0]->count() / outer_num_;

    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, num_labels);

        std::vector<std::pair<Dtype, int> > bottom_data_vector;
        for (int k = 0; k < num_labels; ++k) {
          bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
        }
        std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin(),
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());

        if (bottom_data_vector[0].second == label_value) {
          ++accuracy;
        }

        buffer_data[label_value * num_labels
                    + bottom_data_vector[0].second] += Dtype(1);
      }
    }
  }

  template<typename Dtype>
  string ConfusionMatrixLayer<Dtype>::GetType() { return "ConfusionMatrix"; }

  template<typename Dtype>
  void ConfusionMatrixLayer<Dtype>::PrintConfusionMatrix(bool reset) {
    Dtype *buffer_data = nums_buffer_.mutable_cpu_data();
    stringstream ss;
    ss << "\n\nConfusion Matrix";

    for (int i = 0; i < (num_labels - 2); i++) {
      ss << "\t";
    }
    ss << "| Accuracy\n";

    int sum_j[num_labels];
    caffe_set(num_labels, 0, sum_j);
    for (int i = 0; i < num_labels; i++) {
      int sum_i = 0;
      for (int j = 0; j < num_labels; j++) {
        sum_i += buffer_data[i * num_labels + j];
        ss << int(buffer_data[i * num_labels + j]) << "\t";
        sum_j[j] += buffer_data[i * num_labels + j];
      }
      ss << "| "
         << std::fixed
         << std::setprecision(2)
         << 100.f * buffer_data[i * num_labels + i] / sum_i
         << "%\n";
    }
    int total_samples = 0;
    for (int i = 0; i < num_labels; i++) {
      total_samples += sum_j[i];
      ss << std::fixed
         << std::setprecision(2)
         << 100.f * buffer_data[i * num_labels + i] / sum_j[i]
         << "%\t";
    }
    ss << "| " << total_samples << "\n\n";

    LOG(INFO) << ss.str();

    if (reset)
      caffe_set(num_labels * num_labels, Dtype(0), buffer_data);
  }

  INSTANTIATE_CLASS(ConfusionMatrixLayer);

  REGISTER_LAYER_CLASS(ConfusionMatrix);

}  // namespace caffe
