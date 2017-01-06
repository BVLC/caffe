#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable

__kernel void TEMPLATE(softmax_loss_forward,Dtype)(
    int_tp n, __global const Dtype* prob_data, __global const Dtype* label,
    __global Dtype* loss,
    const int_tp num, const int_tp dim, const int_tp spatial_dim,
    const int has_ignore_label_, const int_tp ignore_label_,
    __global Dtype* counts) {

  for (int_tp index = get_global_id(0); index < n;
      index += get_global_size(0)) {
    const int_tp n = index / spatial_dim;
    const int_tp s = index % spatial_dim;
    const int_tp label_value = (int_tp) (label[n * spatial_dim + s]);
    if (has_ignore_label_ == 1 && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log((Dtype)(
          max((Dtype) (prob_data[n * dim + label_value * spatial_dim + s]),
              (Dtype) FLT_MIN)));
      counts[index] = 1;
    }
  }
}

__kernel void TEMPLATE(softmax_loss_backward,Dtype)(const int_tp nthreads,
                                                    __global const Dtype* top,
                                                    __global const Dtype* label,
                                                    __global Dtype* bottom_diff,
                                                    const int_tp num,
                                                    const int_tp dim,
                                                    const int_tp spatial_dim,
                                                    const int has_ignore_label_,
                                                    const int_tp ignore_label_,
                                                    __global Dtype* counts) {

  const int_tp channels = dim / spatial_dim;

  for (int_tp index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {

    const int_tp n = index / spatial_dim;
    const int_tp s = index % spatial_dim;
    const int_tp label_value = (int_tp) (label[n * spatial_dim + s]);

    if (has_ignore_label_ == 1 && label_value == ignore_label_) {
      for (int_tp c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

// Copied from caffe.pb.h, must keep consistent with the original definition
#if TYPE==TYPE_FLOAT
enum LossParameter_NormalizationMode {
  LossParameter_NormalizationMode_FULL = 0,
  LossParameter_NormalizationMode_VALID = 1,
  LossParameter_NormalizationMode_BATCH_SIZE = 2,
  LossParameter_NormalizationMode_NONE = 3
};
#endif
// Copied from softmax_loss_layer.cpp, must keep consistent with the orignal implementation
Dtype TEMPLATE(get_normalizer, Dtype)(
    enum LossParameter_NormalizationMode normalization_mode, int_tp valid_count,
    int_tp outer_num_, int_tp inner_num_) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = (Dtype)(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = (Dtype)(outer_num_ * inner_num_);
      } else {
        normalizer = (Dtype)(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = (Dtype)(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = (Dtype)(1);
      break;
    default:
      normalizer = (Dtype)(0);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return fmax((Dtype)(1.0), normalizer);
}

Dtype TEMPLATE(asum, Dtype)(int_tp n, __global const Dtype *data, __local Dtype *sum_tmp) {
  Dtype sum = 0;
  for(int_tp i = get_global_id(0); i < n; i += get_global_size(0)) {
    sum += data[i];
  }
  sum = sub_group_reduce_add(sum);
  sum_tmp[get_sub_group_id()] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_sub_group_id() == 0)
    sum = sub_group_reduce_add(sum_tmp[get_sub_group_local_id()]);
  return sum;
}

__kernel void TEMPLATE(softmax_loss_forward_asum, Dtype)(
    int_tp n, int_tp outer_num_, int_tp inner_num_,
    int_tp compute_count_sum, int_tp normalization_type,
    __global const Dtype *loss,
    __global const Dtype *counts, __global Dtype *out) {
    __local Dtype sum_tmp[16];

    Dtype loss_sum = TEMPLATE(asum, Dtype)(n, loss, sum_tmp);
    Dtype counts_sum = -1;
    if (compute_count_sum)
      counts_sum = TEMPLATE(asum, Dtype)(n, counts, sum_tmp);

    if (get_global_id(0) == 0)
      out[0] = loss_sum / TEMPLATE(get_normalizer, Dtype)(normalization_type, counts_sum, outer_num_, inner_num_);
}

#endif
