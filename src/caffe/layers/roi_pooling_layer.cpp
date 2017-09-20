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

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - 2;

  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  if (num_spatial_axes_ == 2) {
    if (roi_pool_param.pooled_size_size()) {
      CHECK(roi_pool_param.pooled_size_size() == 1 || roi_pool_param.pooled_size_size() == 2)
        << "pooled_size must be specified once, or 2 values for Height and Width";
      if (roi_pool_param.pooled_size_size() == 1) {
         pooled_h_ = pooled_w_ = roi_pool_param.pooled_size(0);
      } else {
         pooled_h_ = roi_pool_param.pooled_size(0);
         pooled_w_ = roi_pool_param.pooled_size(1);
      }
    } else {
      pooled_h_ = roi_pool_param.pooled_h();
      pooled_w_ = roi_pool_param.pooled_w();
    }
    CHECK_GT(pooled_h_, 0) << "pooled_h must be > 0";
    CHECK_GT(pooled_w_, 0) << "pooled_w must be > 0";
  } else if (num_spatial_axes_ == 3) {
    if (roi_pool_param.has_pooled_h() || roi_pool_param.has_pooled_w()) {
      CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D pooling.";
      CHECK_EQ(0, roi_pool_param.pooled_size_size())
        << "Either kernel_size or kernel_h/w should be specified, not both.";
      pooled_h_ = roi_pool_param.pooled_h();
      pooled_w_ = roi_pool_param.pooled_w();
    } else {
      const int num_dims = roi_pool_param.pooled_size_size();
      CHECK(num_dims == 1 || num_dims == num_spatial_axes_)
        << "pooled_size must be specified once, or once per spatial dimension"
        << " (pooled_size specified " << num_dims << " times "
        << num_spatial_axes_ << " spatial dims).";
      if (num_dims == 1) {
        pooled_d_ = pooled_h_ = pooled_w_ = roi_pool_param.pooled_size(0);
      } else {
        pooled_d_ = roi_pool_param.pooled_size(0);
        pooled_h_ = roi_pool_param.pooled_size(1);
        pooled_w_ = roi_pool_param.pooled_size(2);
      }

      CHECK_GT(pooled_d_, 0) << "pooled_d must be > 0";
      CHECK_GT(pooled_h_, 0) << "pooled_h must be > 0";
      CHECK_GT(pooled_w_, 0) << "pooled_w must be > 0";
    }
  } else {
    NOT_IMPLEMENTED;
  }

  spatial_scale_ = roi_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->shape(1);
  if (num_spatial_axes_ == 2) {
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(bottom[1]->shape(0), channels_, pooled_h_, pooled_w_);
    max_idx_.Reshape(bottom[1]->shape(0), channels_, pooled_h_, pooled_w_);
  } else {
    depth_ = bottom[0]->shape(2);
    height_ = bottom[0]->shape(3);
    width_ = bottom[0]->shape(4);
    vector<int> pooled_shape(bottom[0]->num_axes(), 0);
    pooled_shape[0] = bottom[1]->shape(0);
    pooled_shape[1] = channels_;
    pooled_shape[2] = pooled_d_;
    pooled_shape[3] = pooled_h_;
    pooled_shape[4] = pooled_w_;
    top[0]->Reshape(pooled_shape);
    max_idx_.Reshape(pooled_shape);
  }
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->shape(0);

  int batch_size = bottom[0]->shape(0);
  size_t top_count = top[0]->count();

  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  if (num_spatial_axes_ == 2) {
    int roi_offset = bottom[1]->offset(1);
    size_t top_offset = top[0]->offset(1);
    // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int n = 0; n < num_rois; ++n) {
      Dtype* cur_top = top_data + n * top_offset;
      int* cur_argmax = argmax_data + n * top_offset;
      const Dtype* roi = bottom_rois + n * roi_offset;
      int roi_batch_ind = roi[0];
      int roi_start_w = round(roi[1] * spatial_scale_);
      int roi_start_h = round(roi[2] * spatial_scale_);
      int roi_end_w = round(roi[3] * spatial_scale_);
      int roi_end_h = round(roi[4] * spatial_scale_);
      CHECK_GE(roi_batch_ind, 0);
      CHECK_LT(roi_batch_ind, batch_size);

      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      const Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_h_);
      const Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_w_);

      const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_h_; ++ph) {
          for (int pw = 0; pw < pooled_w_; ++pw) {
            // Compute pooling region for this output unit:
            //  start (included) = floor(ph * roi_height / pooled_h_)
            //  end (excluded) = ceil((ph + 1) * roi_height / pooled_h_)
            int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
            int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
            int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
            int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));

            hstart = min(max(hstart + roi_start_h, 0), height_);
            hend = min(max(hend + roi_start_h, 0), height_);
            wstart = min(max(wstart + roi_start_w, 0), width_);
            wend = min(max(wend + roi_start_w, 0), width_);

            bool is_empty = (hend <= hstart) || (wend <= wstart);

            const int pool_index = ph * pooled_w_ + pw;
            if (is_empty) {
              cur_top[pool_index] = 0;
              cur_argmax[pool_index] = -1;
              continue;
            }

            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (batch_data[index] > cur_top[pool_index]) {
                  cur_top[pool_index] = batch_data[index];
                  cur_argmax[pool_index] = index;
                }
              }
            }
          }
        }

        // Increment all data pointers by one channel
        batch_data += bottom[0]->offset(0, 1);
        cur_top += top[0]->offset(0, 1);
        cur_argmax += max_idx_.offset(0, 1);
      }
    }
  } else if (num_spatial_axes_ == 3) {
    vector<int> roi_offset_vec(1, 0);
    roi_offset_vec[0] = 1;
    int roi_offset = bottom[1]->offset(roi_offset_vec);

    vector<int> top_offset_vec(1, 0);
    top_offset_vec[0] = 1;
    size_t top_offset = top[0]->offset(top_offset_vec);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int n = 0; n < num_rois; ++n) {
      Dtype* cur_top = top_data + n * top_offset;
      int* cur_argmax = argmax_data + n * top_offset;
      const Dtype* roi = bottom_rois + n * roi_offset;
      int roi_batch_ind = roi[0];
      int roi_start_d = round(roi[1] * spatial_scale_);
      int roi_start_w = round(roi[2] * spatial_scale_);
      int roi_start_h = round(roi[3] * spatial_scale_);
      int roi_end_d = round(roi[4] * spatial_scale_);
      int roi_end_w = round(roi[5] * spatial_scale_);
      int roi_end_h = round(roi[6] * spatial_scale_);
      CHECK_GE(roi_batch_ind, 0);
      CHECK_LT(roi_batch_ind, batch_size);

      int roi_depth = max(roi_end_d - roi_start_d + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      const Dtype bin_size_d = static_cast<Dtype>(roi_depth) / static_cast<Dtype>(pooled_d_);
      const Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_h_);
      const Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_w_);

      vector<int> roi_batch_ind_offset(1, 0);
      roi_batch_ind_offset[0] = roi_batch_ind;
      const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind_offset);

      for (int c = 0; c < channels_; ++c) {
        for (int pd = 0; pd < pooled_d_; ++pd) {
          for (int ph = 0; ph < pooled_h_; ++ph) {
            for (int pw = 0; pw < pooled_w_; ++pw) {
              int dstart = static_cast<int>(floor(static_cast<Dtype>(pd) * bin_size_d));
              int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
              int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
              int dend = static_cast<int>(ceil(static_cast<Dtype>(pd + 1) * bin_size_d));
              int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
              int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));

              dstart = min(max(dstart + roi_start_d, 0), depth_);
              dend = min(max(dend + roi_start_d, 0), depth_);
              hstart = min(max(hstart + roi_start_h, 0), height_);
              hend = min(max(hend + roi_start_h, 0), height_);
              wstart = min(max(wstart + roi_start_w, 0), width_);
              wend = min(max(wend + roi_start_w, 0), width_);

              bool is_empty = (dend <= dstart) || (hend <= hstart) || (wend <= wstart);

              const int pool_index = (pd * pooled_h_ + ph) * pooled_w_ + pw;
              if (is_empty) {
                cur_top[pool_index] = 0;
                cur_argmax[pool_index] = -1;
                continue;
              }

              for (int d = dstart; d < dend; ++d) {
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    const int index = (d * height_ + h) * width_ + w;
                    if (batch_data[index] > cur_top[pool_index]) {
                       cur_top[pool_index] = batch_data[index];
                       cur_argmax[pool_index] = index;
                    }
                  }
                }
              }
            }
          }
        }

        // Increment all data pointers by one channel
        vector<int> offset_vec(2, 0);
        offset_vec[1] = 1;
        batch_data += bottom[0]->offset(offset_vec);
        cur_top += top[0]->offset(offset_vec);
        cur_argmax += max_idx_.offset(offset_vec);
      }
    }
  }
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* bottom_rois = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.cpu_data();
  const int num_rois = top[0]->shape(0);

  if (num_spatial_axes_ == 2) {
    // Accumulate gradient over all ROIs
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      int roi_batch_ind = bottom_rois[roi_n * 5];
      // Accumulate gradients over each bin in this ROI
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_h_; ++ph) {
          for (int pw = 0; pw < pooled_w_; ++pw) {
            int offset_top = ((roi_n * channels_ + c) * pooled_h_ + ph) * pooled_w_ + pw;
            int argmax_index = argmax_data[offset_top];
            if (argmax_index >= 0) {
              int offset_bottom = (roi_batch_ind * channels_ + c) * height_ * width_ + argmax_index;
              bottom_diff[offset_bottom] += top_diff[offset_top];
            }
          }
        }
      }
    }
  } else if (num_spatial_axes_ == 3) {
     // Accumulate gradient over all ROIs
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      int roi_batch_ind = bottom_rois[roi_n * 7];
      // Accumulate gradients over each bin in this ROI
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int c = 0; c < channels_; ++c) {
        for (int pd = 0; pd < pooled_d_; ++pd) {
          for (int ph = 0; ph < pooled_h_; ++ph) {
            for (int pw = 0; pw < pooled_w_; ++pw) {
              int offset_top = (((roi_n * channels_ + c) * pooled_d_ + pd) * pooled_h_ + ph) * pooled_w_ + pw;
              int argmax_index = argmax_data[offset_top];
              if (argmax_index >= 0) {
                int offset_bottom = (roi_batch_ind * channels_ + c) * depth_ * height_ * width_ + argmax_index;
                bottom_diff[offset_bottom] += top_diff[offset_top];
              }
            }
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ROIPoolingLayer);
#endif

INSTANTIATE_CLASS(ROIPoolingLayer);
REGISTER_LAYER_CLASS(ROIPooling);

}  // namespace caffe
