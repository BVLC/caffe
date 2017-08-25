#include <algorithm>
#include <utility>
#include <vector>

#include "caffe/layers/batch_reindex_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
__global__ void BRForward(const int count, const int inner_dim, const Dtype* in,
                          const Dtype* permut, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / (inner_dim);
    int in_n = static_cast<int>(permut[n]);
    out[index] = in[in_n * (inner_dim) + index % (inner_dim)];
  }
}

template<typename Dtype>
void BatchReindexLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  check_batch_reindex(bottom[0]->shape(0), bottom[1]->count(),
                      bottom[1]->cpu_data());
  if (top[0]->count() == 0) {
    return;
  }
  int threads = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  BRForward<Dtype> <<<CAFFE_GET_BLOCKS(threads), CAFFE_CUDA_NUM_THREADS>>>(
      top[0]->count(), bottom[0]->count() / bottom[0]->shape(0),
      bottom[0]->gpu_data(), bottom[1]->gpu_data(), top[0]->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
__global__ void BRBackward(const int count, const int inner_dim,
                           const Dtype* in, const Dtype* top_indexes,
                           const Dtype* begins, const Dtype* counts,
                           Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / (inner_dim);
    out[index] = 0;
    int lower = static_cast<int>(begins[n]);
    int upper = lower + static_cast<int>(counts[n]);
    for (int i = lower; i < upper; ++i) {
      int in_n = static_cast<int>(top_indexes[i]);
      out[index] += in[in_n * (inner_dim) + index % (inner_dim)];
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(BatchReindexLayer);

}  // namespace caffe
