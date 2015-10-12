#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(merge_copy_forward, Dtype)(const int nthreads,
                                                  const int dims,
                                                  __global const Dtype* bottom_a,
                                                  const int forward_a,
                                                  __global const Dtype* bottom_b,
                                                  const int forward_b,
                                                  __global Dtype* top,
                                                  const int num,
                                                  const int channels_a,
                                                  const int channels_b,
                                                  __global const int* shape_a,
                                                  __global const int* shape_b) {
  int pad[6];
  int tmp_idx[6];
  int size_a = 1;
  int size_b = 1;

  for (int i = 0; i < dims; ++i) {
    pad[i] = (shape_b[i] - shape_a[i]) / 2;
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    int batch_id = index / ((channels_a + channels_b) * size_a);
    int bottom_id = ((index - batch_id * (channels_a + channels_b) * size_a)
        / (channels_a * size_a)) % 2;
    int counter = index;
    for (int i = dims - 1; i >= 0; --i) {
      tmp_idx[i] = counter % shape_a[i];
      counter /= shape_a[i];
    }

    if (bottom_id == 0) {
      int channel_id = (index / size_a) % channels_a;
      int aidx = batch_id * channels_a + channel_id;
      for (int i = 0; i < dims; ++i) {
        aidx *= shape_a[i];
        aidx += tmp_idx[i];
      }
      top[index] = (forward_a == 1) ? bottom_a[aidx] : 0;
    } else {
      int channel_id = (index / size_a) % channels_b;
      int bidx = (batch_id * channels_b + channel_id) * size_b;
      int btemp = 1;
      for (int i = dims - 1; i >= 0; --i) {
        bidx += btemp * (tmp_idx[i] + pad[i]);
        btemp *= shape_b[i];
      }
      top[index] = (forward_b == 1) ? bottom_b[bidx] : 0;
    }
  }
}

__kernel void TEMPLATE(merge_copy_backward,Dtype)(const int nthreads,
                                                  const int dims,
                                                  __global Dtype* bottom_a,
                                                  const int backward_a,
                                                  __global Dtype* bottom_b,
                                                  const int backward_b,
                                                  __global const Dtype* top,
                                                  const int num,
                                                  const int channels_a,
                                                  const int channels_b,
                                                  __global const int* shape_a,
                                                  __global const int* shape_b) {
  int pad[6];
  int tmp_idx[6];
  int size_a = 1;
  int size_b = 1;

  for (int i = 0; i < dims; ++i) {
    pad[i] = (shape_b[i] - shape_a[i]) / 2;
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    int batch_id = index / ((channels_a + channels_b) * size_a);
    int bottom_id = ((index - batch_id * (channels_a + channels_b) * size_a)
        / (channels_a * size_a)) % 2;
    int counter = index;
    for (int i = dims - 1; i >= 0; --i) {
      tmp_idx[i] = counter % shape_a[i];
      counter /= shape_a[i];
    }

    if (bottom_id == 0) {
      int channel_id = (index / size_a) % channels_a;
      int aidx = batch_id * channels_a + channel_id;
      for (int i = 0; i < dims; ++i) {
        aidx *= shape_a[i];
        aidx += tmp_idx[i];
      }
      bottom_a[aidx] = (backward_a == 1) ? top[index] : 0;
    } else {
      int channel_id = (index / size_a) % channels_b;
      int bidx = (batch_id * channels_b + channel_id) * size_b;
      int btemp = 1;
      for (int i = dims - 1; i >= 0; --i) {
        bidx += btemp * (tmp_idx[i] + pad[i]);
        btemp *= shape_b[i];
      }
      bottom_b[bidx] = (backward_b == 1) ? top[index] : 0;
    }
  }
}
