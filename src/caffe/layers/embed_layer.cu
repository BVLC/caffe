#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#ifdef USE_CUDA
#include "caffe/util/gpu_util.cuh"
#endif  // USE_CUDA
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif


namespace caffe {

#ifdef USE_CUDA
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
__global__ void EmbedBackward(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_diff, const int M, const int N, const int K,
    Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(top_index, nthreads) {
    const int n = top_index / N;
    const int d = top_index % N;
    const int index = static_cast<int>(bottom_data[n]);
    const int weight_index = index * N + d;
    caffe_gpu_atomic_add(top_diff[top_index], weight_diff + weight_index);
  }
}
#endif  // USE_CUDA

template <typename Dtype>
void EmbedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int count = top[0]->count();

  if(this->device_context()->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA

    EmbedForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    CUDA_KERNEL(CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS)(
        count, bottom_data, weight, M_, N_, K_, top_data);
    if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, Dtype(1),
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), Dtype(1), top_data);
    }
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      // TODO: Implement OpenCL kernel

    if (bias_term_) {
      greentea_gpu_gemm<Dtype>(this->device_context()->id(), CblasNoTrans,
                               CblasNoTrans, M_, N_, 1, Dtype(1),
                               (cl_mem) (bias_multiplier_.gpu_data()), 0,
                               (cl_mem) (this->blobs_[1]->gpu_data()), 0,
                               Dtype(1), (cl_mem) top_data, 0);
    }

#endif  // USE_GREENTEA
    }
}

template <typename Dtype>
void EmbedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[0]) << "Can't backpropagate to EmbedLayer input.";
  if (this->param_propagate_down_[0]) {
    const int top_count = top[0]->count();
    const int count = this->blobs_[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

    if(this->device_context()->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    EmbedBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        CUDA_KERNEL(CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS)(
        top_count, bottom_data, top_diff, M_, N_, K_, weight_diff);
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      // TODO: Implement OpenCL kernel
#endif  // USE_GREENTEA
    }

  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();

    if(this->device_context()->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, Dtype(1), top_diff,
        bias_multiplier_.gpu_data(), Dtype(1), bias_diff);
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      greentea_gpu_gemv<Dtype>(this->device_context()->id(), CblasTrans, M_, N_,
                               Dtype(1), (cl_mem) top_diff, 0,
                               (cl_mem) (bias_multiplier_.gpu_data()), 0,
                               Dtype(1), (cl_mem) bias_diff, 0);
#endif  // USE_GREENTEA
    }


  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EmbedLayer);

}  // namespace caffe

