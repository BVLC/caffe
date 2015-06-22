#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(max_pool_forward_sk,Dtype)(const int nthreads,
__global Dtype* bottom_data,
                                                  const int num,
                                                  const int channels,
                                                  const int height,
                                                  const int width,
                                                  const int pooled_height,
                                                  const int pooled_width,
                                                  const int kernel_h,
                                                  const int kernel_w,
                                                  const int ext_kernel_h,
                                                  const int ext_kernel_w,
                                                  const int stride_h,
                                                  const int stride_w,
                                                  const int kstride_h,
                                                  const int kstride_w,
                                                  const int pad_h,
                                                  const int pad_w,
                                                  __global Dtype* top_data,
                                                  const int use_mask,
                                                  __global int* mask,
                                                  __global Dtype* top_mask) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + ext_kernel_h, height);
    int wend = min(wstart + ext_kernel_w, width);
    hstart = max(hstart, (int) 0);
    wstart = max(wstart, (int) 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    __global Dtype* bottom_data_ptr = bottom_data
        + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += kstride_h) {
      for (int w = wstart; w < wend; w += kstride_w) {
        if (bottom_data_ptr[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data_ptr[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (use_mask == 1) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

__kernel void TEMPLATE(max_pool_backward_sk,Dtype)(
    const int nthreads, __global const Dtype* top_diff, const int use_mask,
    __global const int* mask, __global const Dtype* top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int ext_kernel_h, const int ext_kernel_w,
    const int stride_h, const int stride_w, const int kstride_h,
    const int kstride_w, const int pad_h, const int pad_w,
    __global Dtype* bottom_diff) {

  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {

    __global const int* mask_ptr = mask;
    __global const Dtype* top_diff_ptr = top_diff;

// find out the local index
// find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int pooled_height_1 = pooled_height - 1;
    int pooled_width_1 = pooled_width - 1;
    int phstart = (h < ext_kernel_h) ? h % kstride_h : (h - ext_kernel_h) + 1;
    int phend =
        (h >= pooled_height) ?
            pooled_height_1 - (pooled_height_1 - phstart) % kstride_h : h;
    int pwstart = (w < ext_kernel_w) ? w % kstride_w : (w - ext_kernel_w) + 1;
    int pwend =
        (w >= pooled_width) ?
            pooled_width_1 - (pooled_width_1 - pwstart) % kstride_w : w;

    Dtype gradient = 0;
    int offset = (n * channels + c) * pooled_height * pooled_width;
    top_diff_ptr += offset;
    if (use_mask == 1) {
      mask_ptr += offset;
      for (int ph = phstart; ph <= phend; ph += kstride_h) {
        for (int pw = pwstart; pw <= pwend; pw += kstride_w) {
          if (mask_ptr[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_ptr[ph * pooled_width + pw];
          }
        }
      }
    } else {
      for (int ph = phstart; ph <= phend; ph += kstride_h) {
        for (int pw = pwstart; pw <= pwend; pw += kstride_w) {
          if (top_mask[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_ptr[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

__kernel void TEMPLATE(ave_pool_forward_sk,Dtype)(
    const int nthreads, __global const Dtype* bottom_data, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int ext_kernel_h, const int ext_kernel_w,
    const int stride_h, const int stride_w, const int kstride_h,
    const int kstride_w, const int pad_h, const int pad_w,
    __global Dtype* top_data) {

  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {

    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + ext_kernel_h, height + pad_h);
    int wend = min(wstart + ext_kernel_w, width + pad_w);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    __global const Dtype* bottom_data_ptr = bottom_data;
    bottom_data_ptr += (n * channels + c) * height * width;
    int pool_size = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_data_ptr[h * width + w];
        ++pool_size;
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

__kernel void TEMPLATE(sto_pool_forward_train_sk,Dtype)(
    const int nthreads, __global const Dtype* bottom_data, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int ext_kernel_h, const int ext_kernel_w,
    const int stride_h, const int stride_w, const int kstride_h,
    const int kstride_w, __global Dtype* rand_idx,
    __global Dtype* top_data) {

  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + ext_kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + ext_kernel_w, width);
    Dtype cumsum = 0.;
    __global const Dtype* bottom_data_ptr = bottom_data;
    bottom_data_ptr += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; h += kstride_h) {
      for (int w = wstart; w < wend; w += kstride_w) {
        cumsum += bottom_data_ptr[h * width + w];
      }
    }
    float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; h += kstride_h) {
      for (int w = wstart; w < wend; w += kstride_w) {
        cumsum += bottom_data_ptr[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_data_ptr[h * width + w];
          h = hend;
          w = wend;
        }
      }
    }
  }
}

__kernel void TEMPLATE(sto_pool_forward_test_sk,Dtype)(
    const int nthreads, __global const Dtype* bottom_data, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int ext_kernel_h, const int ext_kernel_w,
    const int stride_h, const int stride_w, const int kstride_h,
    const int kstride_w,
    __global Dtype* top_data) {

  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + ext_kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + ext_kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    __global const Dtype* bottom_data_ptr = bottom_data;
    bottom_data_ptr += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; h += kstride_h) {
      for (int w = wstart; w < wend; w += kstride_w) {
        cumsum += bottom_data_ptr[h * width + w];
        cumvalues += bottom_data_ptr[h * width + w]
            * bottom_data_ptr[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }

}
