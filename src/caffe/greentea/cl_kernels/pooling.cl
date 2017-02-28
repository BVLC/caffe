#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(max_pool_forward,Dtype)(
    const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,
    const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w, const int_tp pad_h,
    const int_tp pad_w,
    __global Dtype* top_data,
    const int use_mask, __global int_tp* mask, __global Dtype* top_mask) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp pw = index % pooled_width;
    const int_tp ph = (index / pooled_width) % pooled_height;
    const int_tp c = (index / pooled_width / pooled_height) % channels;
    const int_tp n = index / pooled_width / pooled_height / channels;
    int_tp hstart = ph * stride_h - pad_h;
    int_tp wstart = pw * stride_w - pad_w;
    const int_tp hend = min(hstart + kernel_h, height);
    const int_tp wend = min(wstart + kernel_w, width);
    hstart = max(hstart, (int_tp)0);
    wstart = max(wstart, (int_tp)0);
    Dtype maxval = -FLT_MAX;
    int_tp maxidx = -1;
    __global const Dtype* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
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

__kernel void TEMPLATE(ave_pool_forward,Dtype)(
    const int_tp nthreads, __global const Dtype* const bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,
    const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w, const int_tp pad_h,
    const int_tp pad_w, __global Dtype* top_data) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    {
      const int_tp pw = index % pooled_width;
      const int_tp ph = (index / pooled_width) % pooled_height;
      const int_tp c = (index / pooled_width / pooled_height) % channels;
      const int_tp n = index / pooled_width / pooled_height / channels;
      int_tp hstart = ph * stride_h - pad_h;
      int_tp wstart = pw * stride_w - pad_w;
      int_tp hend = min(hstart + kernel_h, height + pad_h);
      int_tp wend = min(wstart + kernel_w, width + pad_w);
      const int_tp pool_size = (hend - hstart) * (wend - wstart);
      hstart = max(hstart, (int_tp)0);
      wstart = max(wstart, (int_tp)0);
      hend = min(hend, height);
      wend = min(wend, width);
      Dtype aveval = 0;
      __global const Dtype* bottom_slice = bottom_data
          + (n * channels + c) * height * width;
      for (int_tp h = hstart; h < hend; ++h) {
        for (int_tp w = wstart; w < wend; ++w) {
          aveval += bottom_slice[h * width + w];
        }
      }
      top_data[index] = aveval / pool_size;
    }
  }
}

__kernel void TEMPLATE(sto_pool_forward_train,Dtype)(
    const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,
    const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w,
    __global Dtype* rand_idx,
    __global Dtype* top_data) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp pw = index % pooled_width;
    const int_tp ph = (index / pooled_width) % pooled_height;
    const int_tp c = (index / pooled_width / pooled_height) % channels;
    const int_tp n = index / pooled_width / pooled_height / channels;
    const int_tp hstart = ph * stride_h;
    const int_tp hend = min(hstart + kernel_h, height);
    const int_tp wstart = pw * stride_w;
    const int_tp wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    __global const Dtype* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    // First pass: get sum
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          h = hend;
          w = wend;
        }
      }
    }
  }
}

__kernel void TEMPLATE(sto_pool_forward_test,Dtype)(
    const int_tp nthreads, __global const Dtype* const bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,
    const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w,
    __global Dtype* top_data) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp pw = index % pooled_width;
    const int_tp ph = (index / pooled_width) % pooled_height;
    const int_tp c = (index / pooled_width / pooled_height) % channels;
    const int_tp n = index / pooled_width / pooled_height / channels;
    const int_tp hstart = ph * stride_h;
    const int_tp hend = min(hstart + kernel_h, height);
    const int_tp wstart = pw * stride_w;
    const int_tp wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    __global const Dtype* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    // First pass: get sum
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}

__kernel void TEMPLATE(max_pool_backward,Dtype)(const int_tp nthreads,
                                                __global const Dtype* top_diff,
                                                const int use_mask,
                                                __global const int_tp* mask,
                                                __global const Dtype* top_mask,
                                                const int_tp num,
                                                const int_tp channels,
                                                const int_tp height,
                                                const int_tp width,
                                                const int_tp pooled_height,
                                                const int_tp pooled_width,
                                                const int_tp kernel_h,
                                                const int_tp kernel_w,
                                                const int_tp stride_h,
                                                const int_tp stride_w,
                                                const int_tp pad_h,
                                                const int_tp pad_w,
                                                __global Dtype* bottom_diff) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local index
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp c = (index / width / height) % channels;
    const int_tp n = index / width / height / channels;
    const int_tp phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int_tp phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int_tp pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int_tp pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int_tp offset = (n * channels + c) * pooled_height * pooled_width;
    __global const Dtype* top_diff_slice = top_diff + offset;
    if (use_mask == 1) {
      __global const int_tp* mask_slice = mask + offset;
      for (int_tp ph = phstart; ph < phend; ++ph) {
        for (int_tp pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      __global const Dtype* top_mask_slice = top_mask + offset;
      for (int_tp ph = phstart; ph < phend; ++ph) {
        for (int_tp pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

__kernel void TEMPLATE(ave_pool_backward,Dtype)(const int_tp nthreads,
                                                __global const Dtype* top_diff,
                                                const int_tp num,
                                                const int_tp channels,
                                                const int_tp height,
                                                const int_tp width,
                                                const int_tp pooled_height,
                                                const int_tp pooled_width,
                                                const int_tp kernel_h,
                                                const int_tp kernel_w,
                                                const int_tp stride_h,
                                                const int_tp stride_w,
                                                const int_tp pad_h,
                                                const int_tp pad_w,
                                                __global Dtype* bottom_diff) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local index
    // find out the local offset
    const int_tp w = index % width + pad_w;
    const int_tp h = (index / width) % height + pad_h;
    const int_tp c = (index / width / height) % channels;
    const int_tp n = index / width / height / channels;
    const int_tp phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int_tp phend = min(h / stride_h + 1, pooled_height);
    const int_tp pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int_tp pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0.0;
    __global const Dtype* const top_diff_slice = top_diff
        + (n * channels + c) * pooled_height * pooled_width;
    for (int_tp ph = phstart; ph < phend; ++ph) {
      for (int_tp pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int_tp hstart = ph * stride_h - pad_h;
        int_tp wstart = pw * stride_w - pad_w;
        int_tp hend = min(hstart + kernel_h, height + pad_h);
        int_tp wend = min(wstart + kernel_w, width + pad_w);
        int_tp pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

__kernel void TEMPLATE(sto_pool_backward,Dtype)(
    const int_tp nthreads, __global const Dtype* rand_idx,
    __global const Dtype* const top_diff, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width,
    const int_tp kernel_h, const int_tp kernel_w, const int_tp stride_h,
    const int_tp stride_w, __global Dtype* bottom_diff) {
  for (int_tp index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {
    // find out the local index
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp c = (index / width / height) % channels;
    const int_tp n = index / width / height / channels;
    const int_tp phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int_tp phend = min(h / stride_h + 1, pooled_height);
    const int_tp pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int_tp pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0.0;
    __global const Dtype* rand_idx_slice = rand_idx
        + (n * channels + c) * pooled_height * pooled_width;
    __global const Dtype* top_diff_slice = top_diff
        + (n * channels + c) * pooled_height * pooled_width;
    for (int_tp ph = phstart; ph < phend; ++ph) {
      for (int_tp pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw]
            * (index == (int_tp) (rand_idx_slice[ph * pooled_width + pw])?1.0:0.0);
      }
    }
    bottom_diff[index] = gradient;
  }
}

