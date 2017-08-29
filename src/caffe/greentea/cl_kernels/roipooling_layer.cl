#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

#define OCL_KERNEL_LOOP(i, n)  for (int i = get_global_id(0); i < (n); i += get_global_size(0))

__kernel void TEMPLATE(ROIPoolForward, Dtype)(
    const int nthreads, const __global Dtype* bottom_data,
    const KERNEL_ARG_DTYPE spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    __global Dtype* bottom_rois, __global Dtype* top_data, __global int* argmax_data) {
  OCL_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // The following computation of hstart, wstart, hend, wend is
    // done with integers due to floating precision errors.
    // As the floating point computing on GPU is not identical to CPU,
    // integer computing is used as a workaround.
    // The following approach also works but requires a rigorous analysis:
    // int hstart = (int)(floor(((float)ph * (float)(roi_height)) /
    //                           (float)(pooled_height)));
    // int wstart = (int)(floor(((float)pw * (float)(roi_width)) /
    //                           (float)(pooled_width)));
    // int hend = (int)(ceil(((float)(ph + 1) * (float)(roi_height)) /
    //                        (float)(pooled_height)));
    // int wend = (int)(ceil(((float)(pw + 1) * (float)(roi_width)) /
    //                        (float)(pooled_width)));

    int hstart = (ph * roi_height) / pooled_height;
    if ( (hstart * pooled_height) > (ph * roi_height) ) {
      --hstart;
    }
    int wstart = (pw * roi_width) / pooled_width;
    if ( (wstart * pooled_width) > (pw * roi_width) ) {
      --wstart;
    }
    int hend = ((ph + 1) * roi_height) / pooled_height;
    if ( (hend * pooled_height) < ((ph + 1) * roi_height) ) {
      ++hend;
    }
    int wend = ((pw + 1) * roi_width) / pooled_width;
    if ( (wend * pooled_width) < ((pw + 1) * roi_width) ) {
      ++wend;
    }

    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    float maxval = is_empty ? 0 : -FLT_MAX;
    int maxidx = -1;
    const __global Dtype* input = bottom_data + (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (input[bottom_index] > maxval) {
          maxval = input[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

__kernel void TEMPLATE(ROIPoolBackward, Dtype)(const int nthreads, __global const Dtype* top_diff,
    __global int* argmax_data, const int num_rois, const KERNEL_ARG_DTYPE spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, __global Dtype* bottom_diff,
    __global Dtype* bottom_rois) {
  OCL_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      __global Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
        h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      __global const Dtype* offset_top_diff = top_diff + offset;
      __global int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      float bin_size_h = (float)(roi_height) / (float)(pooled_height);
      float bin_size_w = (float)(roi_width) / (float)(pooled_width);

      int phstart = floor((float)(h - roi_start_h) / bin_size_h);
      int phend   = ceil ((float)(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor((float)(w - roi_start_w) / bin_size_w);
      int pwend   = ceil ((float)(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}
