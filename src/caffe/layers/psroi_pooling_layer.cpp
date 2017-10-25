/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <cfloat>
#include <algorithm>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/psroi_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    PSROIPoolingParameter psroi_pooling_param =
      this->layer_param_.psroi_pooling_param();
    spatial_scale_ = psroi_pooling_param.spatial_scale();
    LOG(ERROR) << "Spatial scale: " << spatial_scale_;

    CHECK_GT(psroi_pooling_param.output_dim(), 0)
      << "output_dim must be > 0";
    CHECK_GT(psroi_pooling_param.group_size(), 0)
      << "group_size must be > 0";

    output_dim_ = psroi_pooling_param.output_dim();
    group_size_ = psroi_pooling_param.group_size();
    pooled_height_ = group_size_;
    pooled_width_ = group_size_;
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // LOG(INFO) << "psroi pooling reshape";
    channels_ = bottom[0]->channels();
    CHECK_EQ(channels_, output_dim_ * group_size_ * group_size_)
      << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(
      bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
    mapping_channel_.Reshape(
      bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
  }


  template <typename Dtype>
  static void PSROIPoolingForward(
    const int num,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim,
    const int group_size,
    Dtype* top_data,
    int* mapping_channel) {
      // LOG(INFO) << "psroi pooling cpu_forward";
      int pixels = width * height;
#ifdef _OPENMP
	#pragma omp parallel for
#endif
     for (int n = 0; n < num; ++n) {
        // per roi

        int roi_add = n * 5;
        // [start, end) interval for spatial sampling
        int roi_batch_ind = bottom_rois[roi_add];
        Dtype roi_start_w =
          static_cast<Dtype>(round(bottom_rois[roi_add + 1])) * spatial_scale;
        Dtype roi_start_h =
          static_cast<Dtype>(round(bottom_rois[roi_add + 2])) * spatial_scale;
        Dtype roi_end_w =
          static_cast<Dtype>(round(bottom_rois[roi_add + 3]) + 1.) * spatial_scale;
        Dtype roi_end_h =
          static_cast<Dtype>(round(bottom_rois[roi_add + 4]) + 1.) * spatial_scale;

        // Force too small ROIs to be 1x1
        Dtype roi_width = max<Dtype>(roi_end_w - roi_start_w, 0.1);  // avoid 0
        Dtype roi_height = max<Dtype>(roi_end_h - roi_start_h, 0.1);

        // Compute w and h at bottom
        Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
        Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

        int top_roi_offset = n * output_dim * pooled_height * pooled_width;
        for (int ctop = 0; ctop < output_dim; ++ctop) {
          // per category
          int top_plane_offset = top_roi_offset + ctop * pooled_height * pooled_width;
          for (int ph = 0; ph < pooled_height; ++ph) {
            int top_row_offset = top_plane_offset + ph * pooled_width;
            for (int pw = 0; pw < pooled_width; ++pw) {
              int index = top_row_offset + pw;
              // The output is in order (n, ctop, ph, pw)
              int hstart = floor(static_cast<Dtype>(ph) * bin_size_h + roi_start_h);
              int wstart = floor(static_cast<Dtype>(pw) * bin_size_w + roi_start_w);
              int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h);
              int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w);
              // Add roi offsets and clip to input boundaries
              hstart = min(max(hstart, 0), height);
              hend = min(max(hend, 0), height);
              wstart = min(max(wstart, 0), width);
              wend = min(max(wend, 0), width);

              bool is_empty = (hend <= hstart) || (wend <= wstart);
              int gw = pw;
              int gh = ph;
              int c = (ctop * group_size + gh) * group_size + gw;

              Dtype out_sum = 0;
              int bottom_base_offset = (roi_batch_ind * channels + c) * pixels;
              const Dtype *current_bottom = bottom_data + bottom_base_offset;
              for (int h = hstart; h < hend; ++h) {
                int roi_row_offset = h * width;
                for (int w = wstart; w < wend; ++w) {
                  int bottom_index = roi_row_offset + w;
                  out_sum += current_bottom[bottom_index];
                }
              }

              Dtype bin_area = (hend - hstart) * (wend - wstart);
              top_data[index] = is_empty ? 0. : out_sum / bin_area;

              mapping_channel[index] = c;
            }
          }
        }
    }
  }


  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_rois = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_cpu_data();
    int count = top[0]->count();
    caffe_set(count, Dtype(0), top_data);
    caffe_set(count, -1, mapping_channel_ptr);
    
    PSROIPoolingForward(bottom[1]->num(), bottom_data, spatial_scale_,
      channels_, height_, width_, pooled_height_,
      pooled_width_, bottom_rois, output_dim_, group_size_,
      top_data, mapping_channel_ptr);
  }

  template <typename Dtype>
    static void PSROIPoolingBackward(
    const int num,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
	// LOG(INFO) << "psroipooling backward cpu";
    int pixels = height * width;
#ifdef _OPENMP
 	#pragma omp parallel for
#endif
    for (int i = 0; i < num; ++i) {
      // The output is in order (n, ctop, ph, pw)
      int pw = i % pooled_width;
      int ph = (i / pooled_width) % pooled_height;
      int n = i / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      int roi_add = n * 5;
      int roi_batch_ind = bottom_rois[roi_add];
      Dtype roi_start_w =
        static_cast<Dtype>(round(bottom_rois[roi_add + 1])) * spatial_scale;
      Dtype roi_start_h =
        static_cast<Dtype>(round(bottom_rois[roi_add + 2])) * spatial_scale;
      Dtype roi_end_w =
        static_cast<Dtype>(round(bottom_rois[roi_add + 3]) + 1.) * spatial_scale;
      Dtype roi_end_h =
        static_cast<Dtype>(round(bottom_rois[roi_add + 4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, (Dtype)0.1);  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, (Dtype)0.1);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph)* bin_size_h + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = mapping_channel[i];
      Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * pixels;
      Dtype bin_area = (hend - hstart) * (wend - wstart);
      Dtype diff_val = is_empty ? 0. : top_diff[i] / bin_area;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          offset_bottom_diff[h * width + w] += diff_val;
        }
      }
    }
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    if (!propagate_down[0]) {
      return;
    }

    const Dtype* bottom_rois = bottom[1]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* mapping_channel_ptr = mapping_channel_.cpu_data();
    caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
    caffe_set(bottom_count, Dtype(0), bottom_diff);
    const int count = top[0]->count();
    PSROIPoolingBackward(count, top_diff, mapping_channel_ptr,
      top[0]->num(), spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, output_dim_, bottom_diff,
      bottom_rois);
  }


#ifdef CPU_ONLY
  STUB_GPU(PSROIPoolingLayer);
#endif

  INSTANTIATE_CLASS(PSROIPoolingLayer);
  REGISTER_LAYER_CLASS(PSROIPooling);

}  // namespace caffe
