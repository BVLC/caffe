#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(max_pool_forward_sk,Dtype)(const int_tp nthreads,
__global Dtype* bottom_data,
                                                  const int_tp num,
                                                  const int_tp channels,
                                                  const int_tp height,
                                                  const int_tp width,
                                                  const int_tp pooled_height,
                                                  const int_tp pooled_width,
                                                  const int_tp kernel_h,
                                                  const int_tp kernel_w,
                                                  const int_tp ext_kernel_h,
                                                  const int_tp ext_kernel_w,
                                                  const int_tp stride_h,
                                                  const int_tp stride_w,
                                                  const int_tp dilation_h,
                                                  const int_tp dilation_w,
                                                  const int_tp pad_h,
                                                  const int_tp pad_w,
                                                  __global Dtype* top_data,
                                                  const int use_mask,
                                                  __global int_tp* mask,
                                                  __global Dtype* top_mask) {
  for (int_tp index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {
    int_tp pw = index % pooled_width;
    int_tp ph = (index / pooled_width) % pooled_height;
    int_tp c = (index / pooled_width / pooled_height) % channels;
    int_tp n = index / pooled_width / pooled_height / channels;
    int_tp hstart = ph * stride_h - pad_h;
    int_tp wstart = pw * stride_w - pad_w;
    int_tp hend = min(hstart + ext_kernel_h, height);
    int_tp wend = min(wstart + ext_kernel_w, width);
    hstart = max(hstart, (int_tp) 0);
    wstart = max(wstart, (int_tp) 0);
    Dtype maxval = -FLT_MAX;
    int_tp maxidx = -1;
    __global Dtype* bottom_data_ptr = bottom_data
        + (n * channels + c) * height * width;
    for (int_tp h = hstart; h < hend; h += dilation_h) {
      for (int_tp w = wstart; w < wend; w += dilation_w) {
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
    const int_tp nthreads, __global const Dtype* top_diff, const int use_mask,
    __global const int_tp* mask, __global const Dtype* top_mask,
    const int_tp num, const int_tp channels, const int_tp height,
    const int_tp width, const int_tp pooled_height, const int_tp pooled_width,
    const int_tp kernel_h, const int_tp kernel_w, const int_tp ext_kernel_h,
    const int_tp ext_kernel_w, const int_tp stride_h, const int_tp stride_w,
    const int_tp dilation_h, const int_tp dilation_w, const int_tp pad_h,
    const int_tp pad_w,
    __global Dtype* bottom_diff) {

  for (int_tp index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {

    __global const int_tp* mask_ptr = mask;
    __global const Dtype* top_diff_ptr = top_diff;

// find out the local index
// find out the local offset
    int_tp w = index % width;
    int_tp h = (index / width) % height;
    int_tp c = (index / width / height) % channels;
    int_tp n = index / width / height / channels;

    int_tp pooled_height_1 = pooled_height - 1;
    int_tp pooled_width_1 = pooled_width - 1;
    int_tp phstart =
        (h < ext_kernel_h) ? h % dilation_h : (h - ext_kernel_h) + 1;
    int_tp phend =
        (h >= pooled_height) ?
            pooled_height_1 - (pooled_height_1 - phstart) % dilation_h : h;
    int_tp pwstart =
        (w < ext_kernel_w) ? w % dilation_w : (w - ext_kernel_w) + 1;
    int_tp pwend =
        (w >= pooled_width) ?
            pooled_width_1 - (pooled_width_1 - pwstart) % dilation_w : w;

    Dtype gradient = 0;
    int_tp offset = (n * channels + c) * pooled_height * pooled_width;
    top_diff_ptr += offset;
    if (use_mask == 1) {
      mask_ptr += offset;
      for (int_tp ph = phstart; ph <= phend; ph += dilation_h) {
        for (int_tp pw = pwstart; pw <= pwend; pw += dilation_w) {
          if (mask_ptr[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_ptr[ph * pooled_width + pw];
          }
        }
      }
    } else {
      for (int_tp ph = phstart; ph <= phend; ph += dilation_h) {
        for (int_tp pw = pwstart; pw <= pwend; pw += dilation_w) {
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
    const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width,
    const int_tp kernel_h, const int_tp kernel_w, const int_tp ext_kernel_h,
    const int_tp ext_kernel_w, const int_tp stride_h, const int_tp stride_w,
    const int_tp dilation_h, const int_tp dilation_w, const int_tp pad_h,
    const int_tp pad_w,
    __global Dtype* top_data) {

  for (int_tp index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {

    int_tp pw = index % pooled_width;
    int_tp ph = (index / pooled_width) % pooled_height;
    int_tp c = (index / pooled_width / pooled_height) % channels;
    int_tp n = index / pooled_width / pooled_height / channels;
    int_tp hstart = ph * stride_h - pad_h;
    int_tp wstart = pw * stride_w - pad_w;
    int_tp hend = min(hstart + ext_kernel_h, height + pad_h);
    int_tp wend = min(wstart + ext_kernel_w, width + pad_w);
    hstart = max(hstart, (int_tp)0);
    wstart = max(wstart, (int_tp)0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    __global const Dtype* bottom_data_ptr = bottom_data;
    bottom_data_ptr += (n * channels + c) * height * width;
    int_tp pool_size = 0;
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        aveval += bottom_data_ptr[h * width + w];
        ++pool_size;
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

__kernel void TEMPLATE(sto_pool_forward_train_sk,Dtype)(
    const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width,
    const int_tp kernel_h, const int_tp kernel_w, const int_tp ext_kernel_h,
    const int_tp ext_kernel_w, const int_tp stride_h, const int_tp stride_w,
    const int_tp dilation_h, const int_tp dilation_w, __global Dtype* rand_idx,
    __global Dtype* top_data) {

  for (int_tp index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {
    int_tp pw = index % pooled_width;
    int_tp ph = (index / pooled_width) % pooled_height;
    int_tp c = (index / pooled_width / pooled_height) % channels;
    int_tp n = index / pooled_width / pooled_height / channels;
    int_tp hstart = ph * stride_h;
    int_tp hend = min(hstart + ext_kernel_h, height);
    int_tp wstart = pw * stride_w;
    int_tp wend = min(wstart + ext_kernel_w, width);
    Dtype cumsum = 0.;
    __global const Dtype* bottom_data_ptr = bottom_data;
    bottom_data_ptr += (n * channels + c) * height * width;
    // First pass: get sum
    for (int_tp h = hstart; h < hend; h += dilation_h) {
      for (int_tp w = wstart; w < wend; w += dilation_w) {
        cumsum += bottom_data_ptr[h * width + w];
      }
    }
    float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int_tp h = hstart; h < hend; h += dilation_h) {
      for (int_tp w = wstart; w < wend; w += dilation_w) {
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
    const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width,
    const int_tp kernel_h, const int_tp kernel_w, const int_tp ext_kernel_h,
    const int_tp ext_kernel_w, const int_tp stride_h, const int_tp stride_w,
    const int_tp dilation_h, const int_tp dilation_w,
    __global Dtype* top_data) {

  for (int_tp index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {
    int_tp pw = index % pooled_width;
    int_tp ph = (index / pooled_width) % pooled_height;
    int_tp c = (index / pooled_width / pooled_height) % channels;
    int_tp n = index / pooled_width / pooled_height / channels;
    int_tp hstart = ph * stride_h;
    int_tp hend = min(hstart + ext_kernel_h, height);
    int_tp wstart = pw * stride_w;
    int_tp wend = min(wstart + ext_kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    __global const Dtype* bottom_data_ptr = bottom_data;
    bottom_data_ptr += (n * channels + c) * height * width;
    // First pass: get sum
    for (int_tp h = hstart; h < hend; h += dilation_h) {
      for (int_tp w = wstart; w < wend; w += dilation_w) {
        cumsum += bottom_data_ptr[h * width + w];
        cumvalues += bottom_data_ptr[h * width + w]
            * bottom_data_ptr[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }

}
