#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/permute_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PermuteKernel(const int nthreads, Dtype *const bottom_data,
			      const int *permute_order,
                              const int *old_steps, const int *new_steps,
                              const int num_axes, Dtype *const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int temp_idx = index;
    int old_idx = 0;
    for (int i = 0; i < num_axes; ++i) {
      int order = permute_order[i];
      old_idx += (temp_idx / new_steps[i]) * old_steps[order];
      temp_idx %= new_steps[i];
    }
    top_data[index] = bottom_data[old_idx];
  }
}

template <typename Dtype>
void PermuteLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  Forward_const_gpu(bottom, top);
}

template <typename Dtype>
void PermuteLayer<Dtype>::Forward_const_gpu(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  if (need_permute_) {
    vector<int> top_shape;
    Blob<int> old_steps;
    Blob<int> new_steps;
    old_steps.Reshape(num_axes_, 1, 1, 1);
    new_steps.Reshape(num_axes_, 1, 1, 1);
    for (int i = 0; i < num_axes_; ++i) {
      if (i == num_axes_ - 1) {
	old_steps.mutable_cpu_data()[i] = 1;
      } else {
	old_steps.mutable_cpu_data()[i] = bottom[0]->count(i + 1);
      }
      top_shape.push_back(bottom[0]->shape(permute_order_.cpu_data()[i]));
    }
    top[0]->Reshape(top_shape);

    for (int i = 0; i < num_axes_; ++i) {
      if (i == num_axes_ - 1) {
	new_steps.mutable_cpu_data()[i] = 1;
      } else {
	new_steps.mutable_cpu_data()[i] = top[0]->count(i + 1);
      }
    }

    Dtype *bottom_data = bottom[0]->mutable_gpu_data();
    Dtype *top_data = top[0]->mutable_gpu_data();
    int count = top[0]->count();
    const int *permute_order = permute_order_.gpu_data();
    const int *new_steps = new_steps.gpu_data();
    const int *old_steps = old_steps.gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, permute_order, old_steps, new_steps,
        num_axes_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    // If there is no need to permute, we share data to save memory.
    top[0]->ShareData(*bottom[0]);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_CONST(PermuteLayer);

} // namespace caffe
