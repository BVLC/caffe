#include <vector>

#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Reverse_gpu(const Dtype* const src,
      Dtype* const dst, const int* const offset_data,
      const int num_reverse_pair, const int reverse_unit_size) {
  CUDA_KERNEL_LOOP(i, num_reverse_pair * reverse_unit_size) {
    const int i1 = i / reverse_unit_size;
    const int i2 = i % reverse_unit_size;
    const int offset_front = offset_data[i1*2]+i2;
    const int offset_back = offset_data[i1*2+1]+i2;
    dst[offset_back] = src[offset_front];
    dst[offset_front] = src[offset_back];
  }
}

template <typename Dtype>
void ReverseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  temp_.CopyFrom(*bottom[0]);
  top[0]->CopyFrom(*bottom[0]);
  if (num_reverse_pairs_ == 0)
    return;
  Reverse_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(num_reverse_pairs_), CAFFE_CUDA_NUM_THREADS>>>(
      temp_.gpu_data(), top[0]->mutable_gpu_data(),
      reverse_offset_.gpu_data(), num_reverse_pairs_, reverse_unit_size_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ReverseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() > 1) {
    CHECK(!propagate_down[1])
        << "Cannot backpropagate to reverse segment Blob";
  }
  temp_.CopyFrom(*top[0], true);
  bottom[0]->CopyFrom(*top[0], true);
  if (num_reverse_pairs_ == 0)
    return;
  Reverse_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(num_reverse_pairs_), CAFFE_CUDA_NUM_THREADS>>>(
      temp_.gpu_diff(), bottom[0]->mutable_gpu_diff(),
      reverse_offset_.gpu_data(), num_reverse_pairs_, reverse_unit_size_);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ReverseLayer);

}  // namespace caffe
