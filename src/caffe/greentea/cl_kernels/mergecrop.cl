#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(merge_copy_forward, Dtype)(
    const int nthreads, __global const Dtype* bottom_a, const int forward_a,
    __global const Dtype* bottom_b, const int forward_b,
    __global Dtype* top,
    int num, int channels_a, int channels_b, int height_a, int width_a,
    int height_b, int width_b) {

  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {

    int pad_h = (height_b - height_a) / 2;
    int pad_w = (width_b - width_a) / 2;

    int batch_id = index / ((channels_a + channels_b) * height_a * width_a);

    int bottom_id = ((index
        - batch_id * (channels_a + channels_b) * height_a * width_a)
        / (channels_a * height_a * width_a)) % 2;

    int h = ((index / width_a) % height_a);
    int w = (index % width_a);

    if (bottom_id == 0) {
      int channel_id = (index / ((width_a * height_a)) % channels_a);
      int aidx = ((((batch_id) * channels_a + channel_id) * height_a + h)
          * width_a + w);
      top[index] = forward_a == 1 ? bottom_a[aidx] : 0;
    } else {
      int channel_id = (index / ((width_a * height_a)) % channels_b);
      int bidx = (((batch_id) * channels_b + channel_id) * height_b * width_b)
          + width_b * (h + pad_h) + pad_w + w;
      top[index] = forward_b == 1 ? bottom_b[bidx] : 0;
    }
  }

}

__kernel void TEMPLATE(merge_copy_backward,Dtype)(const int nthreads,
__global Dtype* bottom_a,
                                                  int backward_a,
                                                  __global Dtype* bottom_b,
                                                  int backward_b,
                                                  __global const Dtype* top,
                                                  int num, int channels_a,
                                                  int channels_b, int height_a,
                                                  int width_a, int height_b,
                                                  int width_b) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    int batch_id = index / ((channels_a + channels_b) * height_a * width_a);

    int pad_h = (height_b - height_a) / 2;
    int pad_w = (width_b - width_a) / 2;

    int bottom_id = ((index
        - batch_id * (channels_a + channels_b) * height_a * width_a)
        / (channels_a * height_a * width_a)) % 2;

    int h = ((index / width_a) % height_a);
    int w = (index % width_a);

    if (bottom_id == 0) {
      int channel_id = (index / ((width_a * height_a)) % channels_a);
      int aidx = ((((batch_id) * channels_a + channel_id) * height_a + h)
          * width_a + w);
      bottom_a[aidx] = backward_a == 1 ? top[index] : 0;
    } else {
      int channel_id = (index / ((width_a * height_a)) % channels_b);
      int bidx = (((batch_id) * channels_b + channel_id) * height_b * width_b)
          + width_b * (h + pad_h) + pad_w + w;
      bottom_b[bidx] = backward_b == 1 ? top[index] : 0;
    }
  }
}
