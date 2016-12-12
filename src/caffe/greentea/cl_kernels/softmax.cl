#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(softmax_forward_slm,Dtype)(const int_tp num, const int_tp channels,
                                   const int_tp spatial_dim,
                                   __global Dtype* scale,
                                   __global const Dtype* data,
                                   __global Dtype* out,
                                   __local Dtype *out_tmp,
                                   __local Dtype *scale_tmp,
                                   __local Dtype *group_tmp) {

  int_tp n = get_global_id(1);
  for (int_tp index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=
      get_global_size(0), ++s) {
    float maxval = -FLT_MAX;
    for (int_tp c = get_global_id(0); c < channels; c += get_global_size(0)) {
      Dtype tmp = data[(n * channels + c) * spatial_dim + s];
      maxval = max((Dtype)tmp, (Dtype)maxval);
    }
    maxval = sub_group_reduce_max(maxval);
    //if (get_sub_group_local_id() == 0)
    group_tmp[get_sub_group_id() * spatial_dim + s] = maxval;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int_tp index = get_global_id(0); index < spatial_dim * get_max_sub_group_size(); index +=
      get_global_size(0)) {
    int_tp s = index / get_max_sub_group_size();
    Dtype maxval = sub_group_reduce_max(group_tmp[get_sub_group_local_id() * spatial_dim + s]);
    //if (get_sub_group_local_id() == 0)
    scale_tmp[s] = maxval;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int_tp index = get_global_id(0); index < channels * spatial_dim;
      index += get_global_size(0)) {
    int_tp s = index % spatial_dim;
    out_tmp[index] = exp(data[n * channels * spatial_dim + index] - scale_tmp[s]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int_tp index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=
      get_global_size(0), ++s) {
    Dtype sum = 0;
    for (int_tp c = get_global_id(0); c < channels; c += get_global_size(0)) {
      sum += out_tmp[c * spatial_dim + s];
    }
    sum = sub_group_reduce_add(sum);
    group_tmp[get_sub_group_id() * spatial_dim + s] = sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int_tp index = get_global_id(0); index < spatial_dim * get_max_sub_group_size(); index +=
      get_global_size(0)) {
    int_tp s = index / get_max_sub_group_size();
    Dtype sum = sub_group_reduce_add(group_tmp[get_sub_group_local_id() * spatial_dim + s]);
    //if (get_sub_group_local_id() == 0)
    scale_tmp[s] = sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int_tp index = get_global_id(0); index < channels * spatial_dim;
      index += get_global_size(0)) {
    int_tp s = index % spatial_dim;
    out[n * channels * spatial_dim + index] = out_tmp[index] / scale_tmp[s];
  }
}

__kernel void TEMPLATE(softmax_forward,Dtype)(const int_tp num, const int_tp channels,
                                   const int_tp spatial_dim,
                                   __global Dtype* scale,
                                   __global const Dtype* data,
                                   __global Dtype* out) {

  int_tp n = get_global_id(1);
  __global Dtype *group_tmp = scale + spatial_dim * num + n * get_max_sub_group_size() * spatial_dim;
  for (int_tp index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=
      get_global_size(0), ++s) {
    float maxval = -FLT_MAX;
    for (int_tp c = get_global_id(0); c < channels; c += get_global_size(0)) {
      Dtype tmp = data[(n * channels + c) * spatial_dim + s];
      maxval = max((Dtype)tmp, (Dtype)maxval);
    }
    maxval = sub_group_reduce_max(maxval);
    //if (get_sub_group_local_id() == 0)
    group_tmp[get_sub_group_id() * spatial_dim + s] = maxval;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int_tp index = get_global_id(0); index < spatial_dim * get_max_sub_group_size(); index +=
      get_global_size(0)) {
    int_tp s = index / get_max_sub_group_size();
    Dtype maxval = sub_group_reduce_max(group_tmp[get_sub_group_local_id() * spatial_dim + s]);
    //if (get_sub_group_local_id() == 0)
    scale[n * spatial_dim + s] = maxval;
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int_tp index = get_global_id(0); index < channels * spatial_dim;
      index += get_global_size(0)) {
    int_tp s = index % spatial_dim;
    out[n * channels * spatial_dim + index] = exp(data[n * channels * spatial_dim + index] - scale[n * spatial_dim + s]);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int_tp index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=
      get_global_size(0), ++s) {
    Dtype sum = 0;
    for (int_tp c = get_global_id(0); c < channels; c += get_global_size(0)) {
      sum += out[n * channels * spatial_dim + c * spatial_dim + s];
    }
    sum = sub_group_reduce_add(sum);
    group_tmp[get_sub_group_id() * spatial_dim + s] = sum;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int_tp index = get_global_id(0); index < spatial_dim * get_max_sub_group_size(); index +=
      get_global_size(0)) {
    int_tp s = index / get_max_sub_group_size();
    Dtype sum = sub_group_reduce_add(group_tmp[get_sub_group_local_id() * spatial_dim + s]);
    //if (get_sub_group_local_id() == 0)
    scale[n * spatial_dim + s] = sum;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int_tp index = get_global_id(0); index < channels * spatial_dim;
      index += get_global_size(0)) {
    int_tp s = index % spatial_dim;
    out[n * channels * spatial_dim + index] /= scale[n * spatial_dim + s];
  }
}
