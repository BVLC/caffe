#include <vector>

#include "caffe/layers/crop_layer.hpp"

namespace caffe {

__device__ int compute_uncropped_index(
    int index,
    const int ndims,
    const int* src_strides,
    const int* dest_strides,
    const int* offsets) {
  int dest_index = index;
  int src_index = 0;
  for (int i = 0; i < ndims; ++i) {
      int coord = dest_index / dest_strides[i];
      dest_index -= coord * dest_strides[i];
      src_index += src_strides[i] * (coord + offsets[i]);
  }
  return src_index;
}

template <typename Dtype>
__global__ void crop_kernel_forward(const int nthreads,
    const int ndims,
    const int* src_strides,
    const int* dest_strides,
    const int* offsets,
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int src_index = compute_uncropped_index(
        index, ndims, src_strides, dest_strides, offsets);
    dest[index] = src[src_index];
  }
}

template <typename Dtype>
__global__ void crop_kernel_backward(const int nthreads,
    const int ndims,
    const int* src_strides,
    const int* dest_strides,
    const int* offsets,
    Dtype* src, const Dtype* dest) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int src_index = compute_uncropped_index(
        index, ndims, src_strides, dest_strides, offsets);
    src[src_index] = dest[index];
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int n = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  crop_kernel_forward<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n,
      bottom[0]->num_axes(),
      src_strides_.gpu_data(),
      dest_strides_.gpu_data(),
      offsets.gpu_data(),
      bottom_data, top_data);
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int n = top[0]->count();

  if (propagate_down[0]) {
    caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    // NOLINT_NEXT_LINE(whitespace/operators)
    crop_kernel_backward<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n,
        bottom[0]->num_axes(),
        src_strides_.gpu_data(),
        dest_strides_.gpu_data(),
        offsets.gpu_data(),
        bottom_diff, top_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CropLayer);

}  // namespace caffe
