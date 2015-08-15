#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CapSequenceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> lengths;
  const int lengths_size
      = this->runtime_param().cap_sequence_param().sequence_lengths_size();
  for (int i = 0; i < lengths_size; ++i) {
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
    caffe_copy(size, bottom[lengths[i]]->gpu_data() + offset,
        top[0]->mutable_gpu_data() + offset);
  }
}

template <typename Dtype>
void CapSequenceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  vector<int> lengths;
  const int lengths_size
      = this->runtime_param().cap_sequence_param().sequence_lengths_size();
  for (int i = 0; i < lengths_size; ++i) {
    lengths.push_back(
        this->runtime_param().cap_sequence_param().sequence_lengths(i));
  }
  int size = 1;
  for (int i = 1; i < bottom[0]->shape().size(); ++i) {
    size *= bottom[0]->shape(i);
  }
  for (int i = 0; i < lengths.size(); ++i) {
    const int offset = i * size;
    caffe_gpu_add(size, bottom[lengths[i]]->gpu_diff() + offset,
        top[0]->gpu_diff() + offset,
        bottom[lengths[i]]->mutable_gpu_diff() + offset);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CapSequenceLayer);

}  // namespace caffe
