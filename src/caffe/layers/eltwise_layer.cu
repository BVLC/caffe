#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data,
    int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_gpu_set(count, Dtype(0.), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    mask = max_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data, mask);
    for (int i = 2; i < bottom.size(); ++i) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_data, bottom[i]->gpu_data(), i-1, top_data, mask);
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
__global__ void MaxBackward(const int nthreads, const Dtype* top_diff,
    const int blob_idx, const int* mask, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe
