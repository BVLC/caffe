#include <cfloat>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ScalarForward(const int n, const Dtype* in,
    const Dtype* scalars, const int scalar_dim, const int inner_dim,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int scalar_index = (index / inner_dim) % scalar_dim;
    out[index] = in[index] * scalars[scalar_index];
  }
}

template <typename Dtype>
void ScalarLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* scalar_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  ScalarForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, scalar_data, scalar_dim_, inner_dim_, top_data);
}

template <typename Dtype>
void ScalarLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Hack: store big eltwise product in bottom[0] diff, except in the special
    // case where this layer itself does the eltwise product, in which case we
    // can store it directly in the scalar diff, and we're done.
    const bool is_eltwise = (inner_dim_ == 1 && outer_dim_ == 1);
    Dtype* product = is_eltwise ?
        bottom[1]->mutable_gpu_diff() : bottom[0]->mutable_gpu_diff();
    caffe_gpu_mul(top[0]->count(), top_diff, bottom_data, product);
    if (!is_eltwise) {
      Dtype* sum_result = NULL;
      if (inner_dim_ == 1) {
        sum_result = product;
      } else if (sum_result_.count() == 1) {
        const Dtype* sum_mult = sum_multiplier_.gpu_data();
        Dtype* scalar_diff = bottom[1]->mutable_cpu_diff();
        caffe_gpu_dot(inner_dim_, product, sum_mult, scalar_diff);
      } else {
        const Dtype* sum_mult = sum_multiplier_.gpu_data();
        sum_result = (outer_dim_ == 1) ?
            bottom[1]->mutable_gpu_diff() : sum_result_.mutable_gpu_data();
        caffe_gpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_,
                       Dtype(1), product, sum_mult, Dtype(0), sum_result);
      }
      if (outer_dim_ != 1) {
        const Dtype* sum_mult = sum_multiplier_.gpu_data();
        if (scalar_dim_ == 1) {
          Dtype* scalar_diff = bottom[1]->mutable_cpu_diff();
          caffe_gpu_dot(outer_dim_, sum_mult, sum_result, scalar_diff);
        } else {
          Dtype* scalar_diff = bottom[1]->mutable_gpu_diff();
          caffe_gpu_gemv(CblasTrans, outer_dim_, scalar_dim_,
                         Dtype(1), sum_result, sum_mult, Dtype(0), scalar_diff);
        }
      }
    }
  }
  if (propagate_down[0]) {
    const int count = top[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* scalar_data = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    ScalarForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, scalar_data, scalar_dim_, inner_dim_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScalarLayer);

}  // namespace caffe
