#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(roi_pool_forward,Dtype)(
    const int_tp nthreads, __global const Dtype* bottom_data, const Dtype spatial_scale,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width, __global Dtype* top_data,
    __global const Dtype* bottom_rois, __global int_tp* argmax_data) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // (n, c, ph, pw) is an element in the pooled output
    int_tp pw = index % pooled_width;
    int_tp ph = (index / pooled_width) % pooled_height;
    int_tp c = (index / pooled_width / pooled_height) % channels;
    int_tp n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int_tp roi_batch_ind = bottom_rois[0];
    int_tp roi_start_w = round(bottom_rois[1] * spatial_scale);
    int_tp roi_start_h = round(bottom_rois[2] * spatial_scale);
    int_tp roi_end_w = round(bottom_rois[3] * spatial_scale);
    int_tp roi_end_h = round(bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int_tp roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int_tp roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = (Dtype)(roi_height)
                       / (Dtype)(pooled_height);
    Dtype bin_size_w = (Dtype)(roi_width)
                       / (Dtype)(pooled_width);

    int_tp hstart = (int_tp)(floor((Dtype)(ph)
                                        * bin_size_h));
    int_tp wstart = (int_tp)(floor((Dtype)(pw)
                                        * bin_size_w));
    int_tp hend = (int_tp)(ceil((Dtype)(ph + 1)
                                     * bin_size_h));
    int_tp wend = (int_tp)(ceil((Dtype)(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int_tp maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        int_tp bottom_index = h * width + w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}


__kernel void TEMPLATE(roi_pool_backward,Dtype)(const int_tp nthreads, __global const Dtype* top_diff, __global const int_tp* argmax_data, const int_tp num_rois, const Dtype spatial_scale, const int_tp channels, const int_tp height,const int_tp width, const int_tp pooled_height, const int_tp pooled_width, __global Dtype* bottom_diff, __global const Dtype* bottom_rois) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // (n, c, h, w) coords in bottom data
    int_tp w = index % width;
    int_tp h = (index / width) % height;
    int_tp c = (index / width / height) % channels;
    int_tp n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int_tp roi_n = 0; roi_n < num_rois; ++roi_n) {
      __global const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int_tp roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int_tp roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      int_tp roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      int_tp roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      int_tp roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int_tp offset = (roi_n * channels + c) * pooled_height * pooled_width;
      __global const Dtype* offset_top_diff = top_diff + offset;
      __global const int_tp* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int_tp roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int_tp roi_height = max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h = (Dtype)(roi_height)
                         / (Dtype)(pooled_height);
      Dtype bin_size_w = (Dtype)(roi_width)
                         / (Dtype)(pooled_width);

      int_tp phstart = floor((Dtype)(h - roi_start_h) / bin_size_h);
      int_tp phend = ceil((Dtype)(h - roi_start_h + 1) / bin_size_h);
      int_tp pwstart = floor((Dtype)(w - roi_start_w) / bin_size_w);
      int_tp pwend = ceil((Dtype)(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int_tp ph = phstart; ph < phend; ++ph) {
        for (int_tp pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}
