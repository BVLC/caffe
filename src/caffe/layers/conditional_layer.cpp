#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConditionalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ConditionalParameter cond_param = this->layer_param_.conditional_param();
  conditional_index_ = cond_param.conditional_index();
  first_reshape_ = true;
  check_threshold_value_ = cond_param.has_threshold_value();
  if (check_threshold_value_)
    threshold_value_ = this->layer_param_.conditional_param().threshold_value();
  output_type_ = cond_param.output_type();
  int max_index = bottom[0]->count()/bottom[0]->num() -1;
  CHECK_LE(conditional_index_, max_index) <<
      "conditional_index_ should be <= bottom[0]->count()/bottom[0]->num() -1";
  CHECK(output_type_ == ConditionalParameter_OUTPUT_TYPE_FILTERED_LABELS
      || output_type_ == ConditionalParameter_OUTPUT_TYPE_FILTERED_INDICES)
      << "output_type must be either FILTERED_INDICES or FILTERED_LABELS";
}

template <typename Dtype>
void ConditionalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom[0] is the "IF_blob"
  // bottom[1] is the "THEN_blob"
  // bottom[2] is the "LABELS_blob"
  // top[0] is the vector of indices that passed the conditional check OR
  //     the vector of labels with indices that passed the conditional check
  // top[1] is the top to which will be forwarded bottom[1]
  const Dtype* bottom_data_IF = bottom[0]->cpu_data();

  int num_items = bottom[0]->num();
  int num_elements = bottom[0]->count()/num_items;
  indices_to_forward_.clear();
  // look through the batch to find who passes the conditional check
  for (size_t item_id = 0; item_id < num_items; ++item_id) {
    int index_IF = item_id*num_elements;
    const Dtype* tmp_data_IF = bottom_data_IF + index_IF;
    const Dtype max_value = *std::max_element(tmp_data_IF,
        tmp_data_IF + num_elements);
    if (*(tmp_data_IF + conditional_index_) == max_value) {
      // if threshold_value is not set, keep the index as good
      if (!check_threshold_value_)
        indices_to_forward_.push_back(item_id);
      // otherwise before adding the index we need to compare the threshold
      // with max_value
      else if (max_value >= threshold_value_)
        indices_to_forward_.push_back(item_id);
    }
  }

  // only items that passed conditional check will be forwarded
  int new_tops_num = indices_to_forward_.size();
  // init
  if (first_reshape_) {
    new_tops_num = bottom[1]->num();
    first_reshape_ = false;
  }

  if (output_type_ == ConditionalParameter_OUTPUT_TYPE_FILTERED_INDICES)
    top[0]->Reshape(new_tops_num, 1, 1, 1);
  else if (output_type_ == ConditionalParameter_OUTPUT_TYPE_FILTERED_LABELS)
    top[0]->Reshape(new_tops_num,
        bottom[2]->channels(),
        bottom[2]->height(),
        bottom[2]->width());
  top[1]->Reshape(new_tops_num,
      bottom[1]->channels(),
      bottom[1]->height(),
      bottom[1]->width());
}

template <typename Dtype>
void ConditionalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data_THEN = bottom[1]->cpu_data();
  const Dtype* bottom_data_LABELS = bottom[2]->cpu_data();

  Dtype* top_data_indices_OR_labels = top[0]->mutable_cpu_data();
  Dtype* top_data = top[1]->mutable_cpu_data();

  int new_tops_num = indices_to_forward_.size();

  // predict phase, we don't need to forward the labels but the indices
  if (output_type_ == ConditionalParameter_OUTPUT_TYPE_FILTERED_INDICES) {
    caffe_copy(new_tops_num, &indices_to_forward_[0],
          top_data_indices_OR_labels);
  }

  size_t size_single_batch = top[1]->count()/top[1]->num();
  size_t size_single_label = bottom[2]->count()/bottom[2]->num();
  for (size_t n = 0; n < new_tops_num; n++) {
    int offset = indices_to_forward_[n];
    int data_offset_top = size_single_batch*n;
    int data_offset_bottom = size_single_batch*offset;

    caffe_copy(size_single_batch, bottom_data_THEN+data_offset_bottom,
        top_data+data_offset_top);

    // train/test phase, we don't need to forward the indices but
    //     the filtered labels
    if (output_type_ == ConditionalParameter_OUTPUT_TYPE_FILTERED_LABELS) {
      int data_offset_top_labels = size_single_label*n;
      int data_offset_bottom_labels = size_single_label*offset;

      caffe_copy(size_single_label,
          bottom_data_LABELS+data_offset_bottom_labels,
          top_data_indices_OR_labels+data_offset_top_labels);
    }
  }
}

template <typename Dtype>
void ConditionalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) { return; }

    const int size_single_batch = top[1]->count()/top[1]->num();
    int index_top = 0;
    std::vector<double> zeros(size_single_batch, 0.0);
    for (size_t n = 0; n < bottom[1]->num(); n++) {
      int offset = indices_to_forward_[n];
      int data_offset_bottom = size_single_batch*n;
      if (n != offset) {  // this data was not been forwarded
        caffe_copy(size_single_batch,
            reinterpret_cast<Dtype*>(&zeros[0]),
            bottom[1]->mutable_cpu_diff() + data_offset_bottom);
      } else {  // this data was been forwarded
        int data_offset_top = size_single_batch*index_top;
        index_top++;
        caffe_copy(size_single_batch,
            top[1]->mutable_cpu_diff() + data_offset_top,
            bottom[1]->mutable_cpu_diff() + data_offset_bottom);
      }
    }
}

#ifdef CPU_ONLY
STUB_GPU(ConditionalLayer);
#endif

INSTANTIATE_CLASS(ConditionalLayer);
REGISTER_LAYER_CLASS(CONDITIONAL, ConditionalLayer);
}  // namespace caffe
