#include <vector>
#include "caffe/layers/transpose_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void transpose_gpu(const int nthreads, const Dtype* from_data,
  Dtype* to_data, const int* from_counts, const int* to_counts,
  const int* map, const int num_axes, int* buf) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int* from_inds = buf + index * num_axes;

    int from_index = index, to_index = 0;
    for (int i = 0; i < num_axes; i++) {
      from_inds[i] = from_index / from_counts[i];
      from_index = from_index % from_counts[i];
    }
  for (int i = 0; i < num_axes; i++) {
    to_index += from_inds[map[i]] * to_counts[i];
  }

  *(to_data + to_index) = *(from_data + index);
  }
}

template <typename Dtype>
void TransposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int nthreads = bottom[0]->count();

  transpose_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom[0]->gpu_data(), top[0]->mutable_gpu_data(),
        bottom_counts_.gpu_data(), top_counts_.gpu_data(),
        forward_map_.gpu_data(), bottom[0]->shape().size(),
        buf_.mutable_gpu_data());
}

template <typename Dtype>
void TransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const int nthreads = bottom[0]->count();

  transpose_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff(),
        top_counts_.gpu_data(), bottom_counts_.gpu_data(),
        backward_map_.gpu_data(), bottom[0]->shape().size(),
        buf_.mutable_gpu_data());
}

INSTANTIATE_LAYER_GPU_FUNCS(TransposeLayer);

}  // namespace caffe
