#include <vector>
#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CapSequenceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ASSERT(this->layer_param_.bottom_size() >= 1,
      "CapSequence must have at least one bottom");
  ASSERT(this->layer_param_.top_size() == 1,
      "CapSequence must have at exactly one top");
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void CapSequenceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> lengths;
  const int num_lengths =
    this->runtime_param().cap_sequence_param().sequence_lengths_size();
  for (int i = 0; i < num_lengths; ++i) {
    lengths.push_back(
        this->runtime_param().cap_sequence_param().sequence_lengths(i));
  }
  int size = 1;
  for (int i = 1; i < bottom[0]->shape().size(); ++i) {
    size *= bottom[0]->shape(i);
  }
  for (int i = 0; i < lengths.size(); ++i) {
    ASSERT(lengths[i] < bottom.size(),
        "Sequence length exceeds the number of bottoms");
    ASSERT(bottom[lengths[i]]->shape() == top[0]->shape(),
        "Cap sequence top and bottoms must be the same shape");
    const int offset = i * size;
    caffe_copy(size, bottom[lengths[i]]->cpu_data() + offset,
        top[0]->mutable_cpu_data() + offset);
  }
}

template <typename Dtype>
void CapSequenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  vector<int> lengths;
  const int num_lengths =
    this->runtime_param().cap_sequence_param().sequence_lengths_size();
  for (int i = 0; i < num_lengths; ++i) {
    lengths.push_back(
        this->runtime_param().cap_sequence_param().sequence_lengths(i));
  }
  int size = 1;
  for (int i = 1; i < bottom[0]->shape().size(); ++i) {
    size *= bottom[0]->shape(i);
  }
  for (int i = 0; i < lengths.size(); ++i) {
    const int offset = i * size;
    caffe_add(size, bottom[lengths[i]]->cpu_diff() + offset,
        top[0]->cpu_diff() + offset,
        bottom[lengths[i]]->mutable_cpu_diff() + offset);
  }
}

#ifdef CPU_ONLY
STUB_GPU(CapSequenceLayer);
#endif

INSTANTIATE_CLASS(CapSequenceLayer);
REGISTER_LAYER_CLASS(CapSequence);

}  // namespace caffe
