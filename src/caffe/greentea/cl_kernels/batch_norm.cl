#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

Dtype TEMPLATE(bn_common,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,
                                const Dtype scale, const Dtype eps,
                                __global const Dtype* mean,
                                __global const Dtype* variance,
                                __global const Dtype* data,
                                int_tp *out_off) {
   const int_tp idx_num = get_global_id(0);
   const int_tp idx_chans = get_global_id(1);
   const int_tp idx_spatial_dim = get_global_id(2);

   Dtype m = mean[idx_chans];
   Dtype v = variance[idx_chans];

   m = -scale * m;
   v = (Dtype)native_powr((Dtype)mad(scale, v, eps), (Dtype)-0.5);

   *out_off = (idx_num * channels + idx_chans) * spatial_dim + idx_spatial_dim;
   return (v * (data[*out_off] + m));
}


__kernel void TEMPLATE(bn_use_global_stats_in_place,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,
                                         const KERNEL_ARG_DTYPE scale, const KERNEL_ARG_DTYPE eps,
                                         __global const Dtype* mean,
                                         __global const Dtype* variance,
                                         __global Dtype* top) {
   int_tp out_off;
   Dtype val = TEMPLATE(bn_common,Dtype)(num, channels, spatial_dim, scale, eps, mean, variance, top, &out_off);
   top[out_off] = val;
}

__kernel void TEMPLATE(bn_use_global_stats_in_place_fused_relu,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,
                                         const KERNEL_ARG_DTYPE scale, const KERNEL_ARG_DTYPE eps,
                                         __global const Dtype* mean,
                                         __global const Dtype* variance,
                                         __global Dtype* top) {
   int_tp out_off;
   Dtype val = TEMPLATE(bn_common,Dtype)(num, channels, spatial_dim, scale, eps, mean, variance, top, &out_off);
   top[out_off] = val > 0.0f ? val : 0.0f;
}

__kernel void TEMPLATE(bn_use_global_stats,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,
                                         const KERNEL_ARG_DTYPE scale, const KERNEL_ARG_DTYPE eps,
                                         __global const Dtype* mean,
                                         __global const Dtype* variance,
                                         __global const Dtype* bottom,
                                         __global Dtype* top) {
   int_tp out_off;
   Dtype val = TEMPLATE(bn_common,Dtype)(num, channels, spatial_dim, scale, eps, mean, variance, bottom, &out_off);
   top[out_off] = val;
}

__kernel void TEMPLATE(bn_use_global_stats_fused_relu,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,
                                         const KERNEL_ARG_DTYPE scale, const KERNEL_ARG_DTYPE eps,
                                         __global const Dtype* mean,
                                         __global const Dtype* variance,
                                         __global const Dtype* bottom,
                                         __global Dtype* top) {
   int_tp out_off;
   Dtype val = TEMPLATE(bn_common,Dtype)(num, channels, spatial_dim, scale, eps, mean, variance, bottom, &out_off);
   top[out_off] =  val > 0.0f ? val : 0.0f;
}
