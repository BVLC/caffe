#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

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
__global__ void CoeffSum(const int count, const int dim,
    const int num_offset, const Dtype coeff, const Dtype* coeff_data,
    const bool backward, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    const int n = num_offset + index / dim;
    const Dtype other_coeff = coeff_data ? coeff_data[n] : Dtype(1);
    const Dtype final_coeff = coeff * other_coeff;
    const Dtype result = in[index] * final_coeff;
    if (num_offset == 0 || backward) {
      out[index] = result;
    } else {
      out[index] += result;
    }
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const int count = top[0]->count();
  const int num = top[0]->num();
  const int dim = count / num;
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* coeff_data = NULL;
  const bool kBackward = false;
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    if (coeff_blob_) {
      coeff_data = bottom[bottom.size() - 1]->gpu_data();
    }
    for (int i = 0; i < bottom.size() - coeff_blob_; ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      CoeffSum<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, dim, i * num, coeffs_[i], coeff_data,
          kBackward, bottom_data, top_data);
      CUDA_POST_KERNEL_CHECK;
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

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const int num = top[0]->num();
  const int dim = count / num;
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* coeff_data = NULL;
  if (coeff_blob_) {
    coeff_data = bottom[bottom.size() - 1]->gpu_data();
  }
  const bool kBackward = true;
  for (int i = 0; i < bottom.size() - coeff_blob_; ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              caffe_copy(count, bottom[j]->gpu_data(), bottom_diff);
              initialized = true;
            } else {
              caffe_gpu_mul(count, bottom[j]->gpu_data(), bottom_diff,
                            bottom_diff);
            }
          }
        } else {
          caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
        }
        caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        CoeffSum<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, dim, i * num, coeffs_[i], coeff_data,
            kBackward, top_diff, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.gpu_data();
        MaxBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, i, mask, bottom_diff);
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe
