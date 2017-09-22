#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/embed_layer.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EmbedForward(const int nthreads, const Dtype* bottom_data,
    const Dtype* weight, const int M, const int N, const int K,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(top_index, nthreads) {
    const int n = top_index / N;
    const int d = top_index % N;
    const int index = static_cast<int>(bottom_data[n]);
    const int weight_index = index * N + d;
    top_data[top_index] = weight[weight_index];
  }
}

template <typename Dtype>
void EmbedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int count = top[0]->count();
  EmbedForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, weight, M_, N_, K_, top_data);
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, Dtype(1),
        bias_multiplier_.gpu_data(),
        this->blobs_[1]->gpu_data(), Dtype(1), top_data);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EmbedLayer);

}  // namespace caffe
