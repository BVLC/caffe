#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(merge_copy_forward_stack, Dtype)(const int_tp nthreads,
                                                  const int_tp dims,
                                                  __global const Dtype* bottom_a,
                                                  const int_tp forward_a,
                                                  __global const Dtype* bottom_b,
                                                  const int_tp forward_b,
                                                  __global Dtype* top,
                                                  const int_tp num,
                                                  const int_tp channels_a,
                                                  const int_tp channels_b,
                                                  __global const int_tp* shape_a,
                                                  __global const int_tp* shape_b) {
  int_tp pad[6];
  int_tp tmp_idx[6];
  int_tp size_a = 1;
  int_tp size_b = 1;

  for (int_tp i = 0; i < dims; ++i) {
    pad[i] = (shape_b[i] - shape_a[i]) / 2;
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    int_tp batch_id = index / ((channels_a + channels_b) * size_a);
    int_tp bottom_id = ((index - batch_id * (channels_a + channels_b) * size_a)
        / (channels_a * size_a)) % 2;
    int_tp counter = index;
    for (int_tp i = dims - 1; i >= 0; --i) {
      tmp_idx[i] = counter % shape_a[i];
      counter /= shape_a[i];
    }

    if (bottom_id == 0) {
      int_tp channel_id = (index / size_a) % channels_a;
      int_tp aidx = batch_id * channels_a + channel_id;
      for (int_tp i = 0; i < dims; ++i) {
        aidx *= shape_a[i];
        aidx += tmp_idx[i];
      }
      top[index] = (forward_a == 1) ? bottom_a[aidx] : 0;
    } else {
      int_tp channel_id = (index / size_a) % channels_b;
      int_tp bidx = (batch_id * channels_b + channel_id) * size_b;
      int_tp btemp = 1;
      for (int_tp i = dims - 1; i >= 0; --i) {
        bidx += btemp * (tmp_idx[i] + pad[i]);
        btemp *= shape_b[i];
      }
      top[index] = (forward_b == 1) ? bottom_b[bidx] : 0;
    }
  }
}

__kernel void TEMPLATE(merge_copy_backward_stack,Dtype)(const int_tp nthreads,
                                                  const int_tp dims,
                                                  __global Dtype* bottom_a,
                                                  const int_tp backward_a,
                                                  __global Dtype* bottom_b,
                                                  const int_tp backward_b,
                                                  __global const Dtype* top,
                                                  const int_tp num,
                                                  const int_tp channels_a,
                                                  const int_tp channels_b,
                                                  __global const int_tp* shape_a,
                                                  __global const int_tp* shape_b) {
  int_tp pad[6];
  int_tp tmp_idx[6];
  int_tp size_a = 1;
  int_tp size_b = 1;

  for (int_tp i = 0; i < dims; ++i) {
    pad[i] = (shape_b[i] - shape_a[i]) / 2;
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  for (int_tp index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {
    int_tp batch_id = index / ((channels_a + channels_b) * size_a);
    int_tp bottom_id = ((index - batch_id * (channels_a + channels_b) * size_a)
        / (channels_a * size_a)) % 2;
    int_tp counter = index;
    for (int_tp i = dims - 1; i >= 0; --i) {
      tmp_idx[i] = counter % shape_a[i];
      counter /= shape_a[i];
    }

    if (bottom_id == 0) {
      int_tp channel_id = (index / size_a) % channels_a;
      int_tp aidx = batch_id * channels_a + channel_id;
      for (int_tp i = 0; i < dims; ++i) {
        aidx *= shape_a[i];
        aidx += tmp_idx[i];
      }
      bottom_a[aidx] = (backward_a == 1) ? top[index] : 0;
    } else {
      int_tp channel_id = (index / size_a) % channels_b;
      int_tp bidx = (batch_id * channels_b + channel_id) * size_b;
      int_tp btemp = 1;
      for (int_tp i = dims - 1; i >= 0; --i) {
        bidx += btemp * (tmp_idx[i] + pad[i]);
        btemp *= shape_b[i];
      }
      bottom_b[bidx] = (backward_b == 1) ? top[index] : 0;
    }
  }
}


__kernel void TEMPLATE(merge_copy_forward_add, Dtype)(const int_tp nthreads,
                                                  const int_tp dims,
                                                  __global const Dtype* bottom_a,
                                                  const int_tp forward_a,
                                                  __global const Dtype* bottom_b,
                                                  const int_tp forward_b,
                                                  __global Dtype* top,
                                                  const int_tp num,
                                                  const int_tp channels,
                                                  __global const int_tp* shape_a,
                                                  __global const int_tp* shape_b) {
  int_tp pad[6];
  int_tp tmp_idx[6];
  int_tp size_a = 1;
  int_tp size_b = 1;

  for (int_tp i = 0; i < dims; ++i) {
    pad[i] = (shape_b[i] - shape_a[i]) / 2;
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  for (int_tp index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {
    int_tp batch_id = index / (channels * size_a);
    int_tp counter = index;
    for (int_tp i = dims - 1; i >= 0; --i) {
      tmp_idx[i] = counter % shape_a[i];
      counter /= shape_a[i];
    }

    top[index] = 0;
    int_tp channel_id = (index / size_a) % channels;
    int_tp aidx = batch_id * channels + channel_id;
    for (int_tp i = 0; i < dims; ++i) {
      aidx *= shape_a[i];
      aidx += tmp_idx[i];
    }
    top[index] = forward_a ? top[index] + bottom_a[aidx] : top[index];
    int_tp bidx = (batch_id * channels + channel_id) * size_b;
    int_tp btemp = 1;
    for (int_tp i = dims - 1; i >= 0; --i) {
      bidx += btemp * (tmp_idx[i] + pad[i]);
      btemp *= shape_b[i];
    }
    top[index] = forward_b ? top[index] + bottom_b[bidx] : top[index];
  }
}

__kernel void TEMPLATE(merge_copy_backward_add,Dtype)(const int_tp nthreads,
                                                  const int_tp dims,
                                                  __global Dtype* bottom_a,
                                                  const int_tp backward_a,
                                                  __global Dtype* bottom_b,
                                                  const int_tp backward_b,
                                                  __global const Dtype* top,
                                                  const int_tp num,
                                                  const int_tp channels,
                                                  __global const int_tp* shape_a,
                                                  __global const int_tp* shape_b) {
  int_tp pad[6];
  int_tp tmp_idx[6];
  int_tp size_a = 1;
  int_tp size_b = 1;

  for (int_tp i = 0; i < dims; ++i) {
    pad[i] = (shape_b[i] - shape_a[i]) / 2;
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  for (int_tp index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {
    int_tp batch_id = index / (channels * size_a);
    int_tp counter = index;
    for (int_tp i = dims - 1; i >= 0; --i) {
      tmp_idx[i] = counter % shape_a[i];
      counter /= shape_a[i];
    }

    int_tp channel_id = (index / size_a) % channels;
    int_tp aidx = batch_id * channels + channel_id;
    for (int_tp i = 0; i < dims; ++i) {
      aidx *= shape_a[i];
      aidx += tmp_idx[i];
    }
    bottom_a[aidx] = backward_a ? top[index] : 0;
    int_tp bidx = (batch_id * channels + channel_id) * size_b;
    int_tp btemp = 1;
    for (int_tp i = dims - 1; i >= 0; --i) {
      bidx += btemp * (tmp_idx[i] + pad[i]);
      btemp *= shape_b[i];
    }
    bottom_b[bidx] = backward_b ? top[index] : 0;
  }
}
