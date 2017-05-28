#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(batch_norm_use_global_stats_in_place,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,
                                         const Dtype scale, const Dtype eps, 
                                         __global const Dtype* mean,
                                         __global const Dtype* variance,
                                         __global Dtype* top) {
   const int_tp idx_num = get_global_id(0);
   const int_tp idx_chans = get_global_id(1);
   const int_tp idx_spatial_dim = get_global_id(2);

   Dtype m = mean[idx_chans];
   Dtype v = variance[idx_chans];

   m = -scale * m;
   v = (Dtype)native_powr((float)mad(scale, v, eps), (float)-0.5);

   const int_tp out_off = (idx_num * channels + idx_chans) * spatial_dim + idx_spatial_dim;
   top[out_off] = v * (top[out_off] + m);
}

__kernel void TEMPLATE(batch_norm_use_global_stats,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,
                                         const Dtype scale, const Dtype eps, 
                                         __global const Dtype* mean,
                                         __global const Dtype* variance,
                                         __global const Dtype* bottom,
                                         __global Dtype* top) {
   const int_tp idx_num = get_global_id(0);
   const int_tp idx_chans = get_global_id(1);
   const int_tp idx_spatial_dim = get_global_id(2);

   Dtype m = mean[idx_chans];
   Dtype v = variance[idx_chans];

   m = -scale * m;
   v = (Dtype)native_powr((float)mad(scale, v, eps), (float)-0.5);

   const int_tp out_off = (idx_num * channels + idx_chans) * spatial_dim + idx_spatial_dim;
   top[out_off] = v * (bottom[out_off] + m);
}
