__kernel void MaxPoolForward(const int count, __global Dtype* bottom_data,
    const int num, const int channels, const int height, const int width,
    const int pooled_h, const int pooled_w, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, __global Dtype* top_data,
    __global int* mask, __global Dtype* top_mask) {
  OCL_KERNEL_LOOP(index, count) {
    int pw = index % pooled_w;
    int ph = (index / pooled_w) % pooled_h;
    int c = (index / pooled_w / pooled_h) % channels;
    int n = index / pooled_w / pooled_h / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

__kernel void AvePoolForward(const int count, __global Dtype* bottom_data,
    const int num, const int channels, const int height, const int width,
    const int pooled_h, const int pooled_w, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, __global Dtype* top_data) {
  OCL_KERNEL_LOOP(index, count) {
    int pw = index % pooled_w;
    int ph = (index / pooled_w) % pooled_h;
    int c = (index / pooled_w / pooled_h) % channels;
    int n = index / pooled_w / pooled_h / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_data[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

__kernel void StoPoolForwardTrain(const int nthreads,
    __global const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, __global Dtype* rand_idx, __global Dtype* top_data) {
  OCL_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.0f;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
      }
    }
    Dtype thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_data[h * width + w];
          return;
        }
      }
    }
  }
}

__kernel void StoPoolForwardTest(const int nthreads,
    __global const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, __global Dtype* top_data) {
  OCL_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.0f;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        cumvalues += bottom_data[h * width + w] * bottom_data[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}

__kernel void MaxPoolBackward(const int count, __global const Dtype* top_diff,
    __global const int* mask, __global const Dtype* top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_h, const int pooled_w, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, __global Dtype* bottom_diff) {
  OCL_KERNEL_LOOP(index, count) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    int phend = min((h + pad_h) / stride_h + 1, pooled_h);
    int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    int pwend = min((w + pad_w) / stride_w + 1, pooled_w);
    Dtype gradient = 0;
    int offset = (n * channels + c) * pooled_h * pooled_w;
    top_diff += offset;
    if (mask) {
      mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask[ph * pooled_w + pw] == h * width + w) {
            gradient += top_diff[ph * pooled_w + pw];
          }
        }
      }
    } else {
      top_mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask[ph * pooled_w + pw] == h * width + w) {
            gradient += top_diff[ph * pooled_w + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

__kernel void AvePoolBackward(const int count, const __global Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_h, const int pooled_w,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    __global Dtype* bottom_diff) {
  OCL_KERNEL_LOOP(index, count) {
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend = min(h / stride_h + 1, pooled_h);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend = min(w / stride_w + 1, pooled_w);
    Dtype gradient = 0;
    top_diff += (n * channels + c) * pooled_h * pooled_w;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff[ph * pooled_w + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

__kernel void StoPoolBackward(const int nthreads,
    __global const Dtype* rand_idx, __global const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, __global Dtype* bottom_diff) {
  OCL_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend = min(h / stride_h + 1, pooled_height);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    rand_idx += (n * channels + c) * pooled_height * pooled_width;
    top_diff += (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff[ph * pooled_width + pw] *
            (index == (int)(rand_idx[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}

