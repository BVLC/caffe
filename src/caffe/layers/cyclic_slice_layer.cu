#include <vector>

#include "caffe/layers/cyclic_slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CyclicSliceForward(const int n,
  const Dtype* bottom_data,
  const int inner_dim, const int channels, const int size,
  Dtype* top_data) {
  // each kernel moves one
  CUDA_KERNEL_LOOP(index, n) {
    const int inner_index = index % inner_dim;
    const int batch_index = index / (channels * inner_dim);
    const int channel_index = (index / inner_dim) % channels;
    const int h = inner_index / size;
    const int w = inner_index % size;
    top_data[ (4*batch_index*channels + channel_index) * inner_dim
      + inner_index] = bottom_data[index];
    top_data[((4*batch_index+1)*channels + channel_index) * inner_dim
      + w*size + (size-1-h)] = bottom_data[index];
    top_data[((4*batch_index+2)*channels + channel_index) * inner_dim
      + (size-1-h)*size + (size-1-w)] = bottom_data[index];
    top_data[((4*batch_index+3)*channels + channel_index) * inner_dim
      + (size-1-w)*size + h] = bottom_data[index];
  }
}



template <typename Dtype>
void CyclicSliceLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int inner_dim = bottom[0]->count(2);
  const int size = bottom[0]->shape(2);
  const int channels = bottom[0]->shape(1);
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  CyclicSliceForward<Dtype> <<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(count,
    bottom_data, inner_dim, channels, size, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(CyclicSliceLayer);

}  // namespace caffe
