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

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  // find channel axis and compute spatial axes constants
  channel_axis_ = bottom[0]->CanonicalAxisIndex(pool_param.axis());
  channels_ = bottom[0]->shape(channel_axis_);
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 1);

  if (num_spatial_axes_ == 2) {
      // Process 2D Pooling
      if (pool_param.global_pooling()) {
        CHECK(!(pool_param.kernel_size_size() ||
          pool_param.has_kernel_h() || pool_param.has_kernel_w()))
          << "With Global_pooling: true Filter size cannot specified";
      } else {
        CHECK(!pool_param.kernel_size_size() !=
          !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
          << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
        CHECK(pool_param.kernel_size_size() ||
          (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
          << "For non-square filters both kernel_h and kernel_w are required.";
      }
      CHECK((!pool_param.pad_size() && pool_param.has_pad_h()
          && pool_param.has_pad_w())
          || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
          << "pad is pad OR pad_h and pad_w are required.";
      CHECK((!pool_param.stride_size() && pool_param.has_stride_h()
          && pool_param.has_stride_w())
          || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
          << "Stride is stride OR stride_h and stride_w are required.";
      global_pooling_ = pool_param.global_pooling();
      if (global_pooling_) {
        kernel_h_ = bottom[0]->height();
        kernel_w_ = bottom[0]->width();
      } else {
        if (pool_param.kernel_size_size()) {
          CHECK(pool_param.kernel_size_size() == 1 || pool_param.kernel_size_size() == 2)
              << "kernel_size must be specified once, or 2 values for Height and Width";
          if (pool_param.kernel_size_size() == 1) {
              kernel_h_ = kernel_w_ = pool_param.kernel_size(0);
          } else {
              kernel_h_ = pool_param.kernel_size(0);
              kernel_w_ = pool_param.kernel_size(1);
          }
        } else {
          kernel_h_ = pool_param.kernel_h();
          kernel_w_ = pool_param.kernel_w();
        }
      }
      CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
      CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
      if (pool_param.pad_size() > 0) {
          CHECK(pool_param.pad_size() == 1 || pool_param.pad_size() == 2)
              << "pad must be specified once, or 2 values for Height and Width";
          if (pool_param.pad_size() == 1) {
              pad_h_ = pad_w_ = pool_param.pad(0);
          } else {
              pad_h_ = pool_param.pad(0);
              pad_w_ = pool_param.pad(1);
          }
        } else {
        pad_h_ = pool_param.pad_h();
        pad_w_ = pool_param.pad_w();
      }
      if (pool_param.stride_size() > 0) {
          CHECK(pool_param.stride_size() == 1 || pool_param.stride_size() == 2)
              << "stride must be specified once, or 2 values for Height and Width";
          if (pool_param.stride_size() == 1) {
              stride_h_ = stride_w_ = pool_param.stride(0);
          } else {
              stride_h_ = pool_param.stride(0);
              stride_w_ = pool_param.stride(1);
          }
        } else {
        stride_h_ = pool_param.stride_h();
        stride_w_ = pool_param.stride_w();
      }
      if (global_pooling_) {
        CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
          << "With Global_pooling: true; only pad = 0 and stride = 1";
      }
      if (pad_h_ != 0 || pad_w_ != 0) {
        CHECK(this->layer_param_.pooling_param().pool()
            == PoolingParameter_PoolMethod_AVE
            || this->layer_param_.pooling_param().pool()
            == PoolingParameter_PoolMethod_MAX)
            << "Padding implemented only for average and max pooling.";
        CHECK_LT(pad_h_, kernel_h_);
        CHECK_LT(pad_w_, kernel_w_);
      }
    } else if (num_spatial_axes_ == 3) {
      // Process 3D Pooling
      // Setup input dimensions (input_shape_).
      vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);

      // LOG(INFO) << "channel_axis_: " << channel_axis_ << "  channels_: " << channels_ << " num_axes: " << num_axes;
      input_shape_.Reshape(bottom_dim_blob_shape);

      int* input_shape_data = input_shape_.mutable_cpu_data();
      for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
        input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
      }
      vector<int> spatial_dim_blob_shape(1, num_spatial_axes_);

      global_pooling_ = pool_param.global_pooling();
      // Setup filter kernel dimensions (kernel_shape_).
      kernel_shape_.Reshape(spatial_dim_blob_shape);
      int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
      if (global_pooling_) {
        // if global pooling height and width are set the entire blob,
        // and the layer cannot have a kernel set
        CHECK_GE(0, pool_param.kernel_size_size())
            << "With Global_pooling: true Filter size cannot specified.";
        CHECK(!pool_param.has_kernel_h() || !pool_param.has_kernel_w())
            << "With Global_pooling: true Filter size cannot specified.";
        for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
          kernel_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
        }
      } else {
         // if kernel_h or kernel_w are set we cannot set the kernel another way
         // And there must be 2 spatial dims
        if (pool_param.has_kernel_h() || pool_param.has_kernel_w()) {
            CHECK_EQ(num_spatial_axes_, 2)
              << "kernel_h & kernel_w can only be used for 2D pooling.";
            CHECK_EQ(0, pool_param.kernel_size_size())
              << "Either kernel_size or kernel_h/w should be specified, not both.";
            kernel_shape_data[0] = pool_param.kernel_h();
            kernel_shape_data[1] = pool_param.kernel_w();
        } else {
            // using repeated kernel param
            const int num_kernel_dims = pool_param.kernel_size_size();
            CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
              << "kernel_size must be specified once, or once per spatial dimension"
              << " (kernel_size specified " << num_kernel_dims << " times "
              << num_spatial_axes_ << " spatial dims).";
            for (int i = 0; i < num_spatial_axes_; ++i) {
                kernel_shape_data[i] = pool_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
            }
        }
      }

      for (int i = 0; i < num_spatial_axes_; ++i) {
          CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
      }

      // setup padding dimensions (pad_)
      pad_.Reshape(spatial_dim_blob_shape);
      int* pad_data = pad_.mutable_cpu_data();
      int pad_sum = 0;
      if (pool_param.has_pad_h() || pool_param.has_pad_w()) {
          // if pad_h or pad_w are set we cannot set the pad another way
          // And there must be 2 spatial dims
          CHECK_EQ(num_spatial_axes_, 2)
            << "pad_h & pad_w can only be used for 2D convolution.";
          CHECK_EQ(0, pool_param.pad_size())
            << "Either pad or pad_h/w should be specified, not both.";
          pad_data[0] = pool_param.pad_h();
          pad_data[1] = pool_param.pad_w();
      } else {
        // using repeated pad param
        const int num_pad_dims = pool_param.pad_size();
        CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
              num_pad_dims == num_spatial_axes_)
            << "pad must be specified once, or once per spatial dimension "
            << "(pad specified " << num_pad_dims << " times "
            << num_spatial_axes_ << " spatial dims).";
        const int kDefaultPad = 0;
        for (int i = 0; i < num_spatial_axes_; ++i) {
          pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
              pool_param.pad((num_pad_dims == 1) ? 0 : i);
          if (global_pooling_) {
              CHECK_EQ(pad_data[i], 0)
                << "With Global_pooling: true; pool = 0";
            }
          CHECK_LT(pad_data[i], kernel_shape_data[i]);
          pad_sum += pad_data[i];
        }
      }

      if (pad_sum != 0) {
         CHECK(this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_AVE
          || this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_MAX)
            << "Padding implemented only for average and max pooling.";
      }

      // Setup stride dimensions (stride_).
      stride_.Reshape(spatial_dim_blob_shape);
      int* stride_data = stride_.mutable_cpu_data();
      if (pool_param.has_stride_h() || pool_param.has_stride_w()) {
        CHECK_EQ(num_spatial_axes_, 2)
            << "stride_h & stride_w can only be used for 2D convolution.";
        CHECK_EQ(0, pool_param.stride_size())
            << "Either stride or stride_h/w should be specified, not both.";
        stride_data[0] = pool_param.stride_h();
        stride_data[1] = pool_param.stride_w();
      } else {
        // using repeated stride param
        const int num_stride_dims = pool_param.stride_size();
        CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
              num_stride_dims == num_spatial_axes_)
            << "stride must be specified once, or once per spatial dimension "
            << "(stride specified " << num_stride_dims << " times "
            << num_spatial_axes_ << " spatial dims).";
        const int kDefaultStride = 1;
        for (int i = 0; i < num_spatial_axes_; ++i) {
          stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
              pool_param.stride((num_stride_dims == 1) ? 0 : i);
          CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
          if (global_pooling_) {
            CHECK_EQ(stride_data[i], 1)
              << "With Global_pooling: true; stride = 1";
          }
        }
      }
    } else {
      NOT_IMPLEMENTED;
    }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(pool_param.axis());
  num_ = bottom[0]->count(0, channel_axis_);
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 1);

  if (num_spatial_axes_ == 2) {
      // Process 2D Pooling
      CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
          << "corresponding to (num, channels, height, width).";
      channels_ = bottom[0]->channels();
      height_ = bottom[0]->height();
      width_ = bottom[0]->width();
      if (global_pooling_) {
        kernel_h_ = bottom[0]->height();
        kernel_w_ = bottom[0]->width();
      }
      pooled_height_ = static_cast<int>(ceil(static_cast<float>(
          height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
      pooled_width_ = static_cast<int>(ceil(static_cast<float>(
          width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
      if (pad_h_ || pad_w_ || kernel_h_ == 1 || kernel_w_ == 1) {
        // If we have padding, ensure that the last pooling starts strictly
        // inside the image (instead of at the padding); otherwise clip the last.
        if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
          --pooled_height_;
        }
        if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
          --pooled_width_;
        }
        CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
        CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
      }
      top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
          pooled_width_);
      if (top.size() > 1) {
        top[1]->ReshapeLike(*top[0]);
      }
      // If max pooling, we will initialize the vector index part.
      if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_MAX && top.size() == 1) {
        max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
            pooled_width_);
      }
      // If stochastic pooling, we will initialize the random index part.
      if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_STOCHASTIC) {
        rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
      }
  } else if (num_spatial_axes_ == 3) {
      /* Process 3D Pooling */
      int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
      const int* input_shape_data = this->input_shape_.cpu_data();
      if (global_pooling_) {
        for (int i = 0; i < num_spatial_axes_; ++i) {
          kernel_shape_data[i] = input_shape_data[i+1];
        }
      }
      // compute output shape
      const int* pad_data = this->pad_.cpu_data();
      const int* stride_data = this->stride_.cpu_data();
      vector<int> spatial_dim_blob_shape(1, num_spatial_axes_);
      output_shape_.Reshape(spatial_dim_blob_shape);
      int* output_shape_data = output_shape_.mutable_cpu_data();
      int pad_sum = 0;
      for (int i = 0; i < num_spatial_axes_; ++i) {
        int oc = static_cast<int>(ceil(static_cast<float>(
              input_shape_data[i+1] + 2 * pad_data[i]
              - kernel_shape_data[i]) / stride_data[i])) + 1;
        pad_sum += pad_data[i];
        output_shape_data[i] = oc;
      }

      if (pad_sum) {
        for (int i = 0; i < num_spatial_axes_; ++i) {
            if ((output_shape_data[i] - 1) * stride_data[i] >=
              input_shape_data[i+1] + pad_data[i])
                --output_shape_data[i];
            CHECK_LT((output_shape_data[i] - 1) * stride_data[i],
              input_shape_data[i+1] + pad_data[i]);
        }
      }

      vector<int> top_shape = bottom[0]->shape();
      // Discard input spatial axes
      top_shape.resize(first_spatial_axis);
      for (int i = 0; i < num_spatial_axes_; ++i) {
          top_shape.push_back(output_shape_data[i]);
      }

      top[0]->Reshape(top_shape);
      if (top.size() > 1) {
        top[1]->ReshapeLike(*top[0]);
      }

      // If max pooling, we will initialize the vector index part.
      if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_MAX && top.size() == 1) {
        max_idx_.Reshape(top_shape);
      }
      // If stochastic pooling, we will initialize the random index part.
      if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_STOCHASTIC) {
        rand_idx_.Reshape(top_shape);
      }
  } else {
    NOT_IMPLEMENTED;
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;

  if (num_spatial_axes_ == 2) {
      /* Process 2D Pooling */
      typename PoolingCodeGeneratorForward<Dtype>::Callback_t* generator_func =
               Forward_code_generator.Get_callback(this, top[0], use_top_mask);
      // We are getting top_mask here as mutable_cpu_data is not thread safe
      // and doing it inside parallel region creates of risk of race condition
      void* mask = NULL;
      if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_MAX ) {
        mask = (use_top_mask) ? static_cast<void*>(top[1]->mutable_cpu_data()) :
                                static_cast<void*>(max_idx_.mutable_cpu_data());
      }

      const int batch_size = bottom[0]->num();
      const int num_channels = bottom[0]->channels();

#ifdef _OPENMP
      #pragma omp parallel for collapse(2)
#endif
      for (int image = 0; image < batch_size; ++image)
        for (int channel = 0; channel < num_channels; ++channel)
          generator_func(bottom_data,
                         top_data,
                         top_count,
                         image,
                         image+1,
                         mask,
                         channel,
                         channel+1,
                         this,
                         use_top_mask);
  } else if (num_spatial_axes_ == 3) {
      // Process 3D Pooling
      vector<int> offset(2, 0);
      offset[1] = 1;

      const int* kernel_shape = kernel_shape_.cpu_data();
      const int* pad_data = this->pad_.cpu_data();
      const int* stride_data = this->stride_.cpu_data();
      const int* input_shape_data = this->input_shape_.cpu_data();
      const int* output_shape_data = this->output_shape_.cpu_data();

      long bottom_offset = bottom[0]->offset(offset);
      long top_offset = top[0]->offset(offset);

      // Different pooling methods. We explicitly do the switch outside the for
      // loop to save time, although this results in more code.
      switch (this->layer_param_.pooling_param().pool()) {
      case PoolingParameter_PoolMethod_MAX:
        if (use_top_mask) {
          caffe_set(top_count, Dtype(-1), top[1]->mutable_cpu_data());
        } else {
          caffe_set(top_count, -1, max_idx_.mutable_cpu_data());
        }
        caffe_set(top_count, Dtype(-FLT_MAX), top_data);

#ifdef _OPENMP
        #pragma omp parallel for collapse(2)
#endif
        for (int n = 0; n < num_; ++n) {
          for (int c = 0; c < channels_; ++c) {
            long nc = n * channels_ + c;
            const Dtype *bottom_data2 = bottom[0]->cpu_data() + nc * bottom_offset;
            Dtype *top_data2 = top[0]->mutable_cpu_data() + nc * top_offset;
            Dtype *top_mask = NULL;
            int *mask = NULL;
            if (use_top_mask) {
              top_mask = top[1]->mutable_cpu_data() + nc * top_offset;
            } else {
              mask = max_idx_.mutable_cpu_data() + nc * top_offset;
            }

            for (int pz = 0; pz < output_shape_data[0]; ++pz) {
              for (int ph = 0; ph < output_shape_data[1]; ++ph) {
                for (int pw = 0; pw < output_shape_data[2]; ++pw) {
                    int zstart = pz * stride_data[0] - pad_data[0];
                    int hstart = ph * stride_data[1] - pad_data[1];
                    int wstart = pw * stride_data[2] - pad_data[2];

                    int zend = min(zstart + kernel_shape[0], input_shape_data[1]);
                    int hend = min(hstart + kernel_shape[1], input_shape_data[2]);
                    int wend = min(wstart + kernel_shape[2], input_shape_data[3]);
 
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    zstart = max(zstart, 0);
                    const int pool_index = (pz * output_shape_data[1] + ph) * output_shape_data[2] + pw;
                    for (int z = zstart; z < zend; ++z) {
                      for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                          const int index = (z * input_shape_data[2] + h) * input_shape_data[3] + w;
                          if (bottom_data2[index] > top_data2[pool_index]) {
                            top_data2[pool_index] = bottom_data2[index];
                            if (use_top_mask) {
                              top_mask[pool_index] = static_cast<Dtype>(index);
                            } else {
                              mask[pool_index] = index;
                            }
                          }
                        }
                      }
                    }
                }
              }
            }
          }
        }
        break;
      case PoolingParameter_PoolMethod_AVE:
        caffe_set(top_count, Dtype(0), top_data);
#ifdef _OPENMP
  #pragma omp parallel for collapse(2)
#endif
        for (int n = 0; n < num_; ++n) {
          for (int c = 0; c < channels_; ++c) {
            long nc = n * channels_ + c;
            const Dtype *bottom_data2 = bottom[0]->cpu_data() + nc * bottom_offset;
            Dtype *top_data2 = top[0]->mutable_cpu_data() + nc * top_offset;

            for (int pz = 0; pz < output_shape_data[0]; ++pz) {
              for (int ph = 0; ph < output_shape_data[1]; ++ph) {
                for (int pw = 0; pw < output_shape_data[2]; ++pw) {
                  int zstart = pz * stride_data[0] - pad_data[0];
                  int hstart = ph * stride_data[1] - pad_data[1];
                  int wstart = pw * stride_data[2] - pad_data[2];

                  int zend = min(zstart + kernel_shape[0],
                           input_shape_data[1] + pad_data[0]);
                  int hend = min(hstart + kernel_shape[1],
                            input_shape_data[2] + pad_data[1]);
                  int wend = min(wstart + kernel_shape[2],
                          input_shape_data[3] + pad_data[2]);

                  int pool_size = (hend - hstart) *
                                  (wend - wstart) *
                                  (zend - zstart);
                  hstart = max(hstart, 0);
                  wstart = max(wstart, 0);
                  zstart = max(zstart, 0);
                  zend = min(zend, input_shape_data[1]);
                  hend = min(hend, input_shape_data[2]);
                  wend = min(wend, input_shape_data[3]);

                  const int pool_index = (pz * output_shape_data[1] + ph) * output_shape_data[2] + pw;
                  for (int z = zstart; z < zend; ++z) {
                    for (int h = hstart; h < hend; ++h) {
                      for (int w = wstart; w < wend; ++w) {
                        const int index = (z * input_shape_data[2] + h) * input_shape_data[3] + w;
                        top_data2[pool_index] += bottom_data2[index];
                      }
                    }
                  }
                  top_data2[pool_index] /= pool_size;
                }
              }
            }
          }
        }
        break;
      case PoolingParameter_PoolMethod_STOCHASTIC:
        NOT_IMPLEMENTED;
        break;
      default:
        LOG(FATAL) << "Unknown pooling method.";
      }
    } else {
      NOT_IMPLEMENTED;
    }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;

  if (num_spatial_axes_ == 2) {
      // Process 2D pooling
      typename PoolingCodeGeneratorBackward<Dtype>::Callback_t* generator_func =
                          Backward_code_generator.Get_callback(this, top[0]);

      // We are getting top_mask here as mutable_cpu_data is not thread safe
      // and doing it inside parallel region creates of risk of race condition
      void* mask = NULL;
      if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_MAX ) {
        mask = (use_top_mask) ? static_cast<void*>(top[1]->mutable_cpu_data()) :
                                static_cast<void*>(max_idx_.mutable_cpu_data());
      }

      const int batch_size = bottom[0]->num();
      const int num_channels = bottom[0]->channels();

#ifdef _OPENMP
      #pragma omp parallel for collapse(2)
#endif
      for (int image = 0; image < batch_size; ++image)
        for (int channel = 0; channel < num_channels; ++channel)
          generator_func(top_diff,
                         bottom_diff,
                         image,
                         image+1,
                         channel,
                         channel+1,
                         use_top_mask,
                         mask,
                         this);
      } else if (num_spatial_axes_ == 3) {
        /* Process 3D pooling */
        const int* kernel_shape = this->kernel_shape_.cpu_data();
        const int* pad_data = this->pad_.cpu_data();
        const int* stride_data = this->stride_.cpu_data();
        const int* input_shape_data = this->input_shape_.cpu_data();
        const int* output_shape_data = this->output_shape_.cpu_data();
        int top_num = top[0]->count(0, channel_axis_);
        vector<int> offset(2, 0);
        offset[1] = 1;

        long bottom_offset = bottom[0]->offset(offset);
        long top_offset = top[0]->offset(offset);

        switch (this->layer_param_.pooling_param().pool()) {
        case PoolingParameter_PoolMethod_MAX:
#ifdef _OPENMP
      #pragma omp parallel for collapse(2)
#endif
          for (int n = 0; n < top_num; ++n) {
            for (int c = 0; c < channels_; ++c) {
              long nc = n * channels_ + c;
              Dtype *bottom_diff2 = bottom[0]->mutable_cpu_diff() + nc * bottom_offset;
              const Dtype *top_diff2 = top[0]->cpu_diff() + nc * top_offset;
              const Dtype *top_mask = NULL;
              const int *mask = NULL;
              if (use_top_mask) {
                top_mask = top[1]->cpu_data() + nc * top_offset;
              } else {
                mask = max_idx_.cpu_data() + nc * top_offset;
              }
 
              for (int pz = 0; pz < output_shape_data[0]; ++pz) {
                for (int ph = 0; ph < output_shape_data[1]; ++ph) {
                  for (int pw = 0; pw < output_shape_data[2]; ++pw) {
                    const int index = (pz * output_shape_data[1] + ph) * output_shape_data[2] + pw;
                    const int bottom_index = use_top_mask ? top_mask[index] : mask[index];
                    bottom_diff2[bottom_index] += top_diff2[index];
                  }
                }
              }
            }
          }
          break;
        case PoolingParameter_PoolMethod_AVE:
#ifdef _OPENMP
  #pragma omp parallel for collapse(2)
#endif
          for (int n = 0; n < top_num; ++n) {
            for (int c = 0; c < channels_; ++c) {
              long nc = n * channels_ + c;
              Dtype *bottom_diff2 = bottom[0]->mutable_cpu_diff() + nc * bottom_offset;
              const Dtype *top_diff2 = top[0]->cpu_diff() + nc * top_offset;

              for (int pz = 0; pz < output_shape_data[0]; ++pz) {
                for (int ph = 0; ph < output_shape_data[1]; ++ph) {
                  for (int pw = 0; pw < output_shape_data[2]; ++pw) {
                    int zstart = pz * stride_data[0] - pad_data[0];
                    int hstart = ph * stride_data[1] - pad_data[1];
                    int wstart = pw * stride_data[2] - pad_data[2];

                    int zend = min(zstart + kernel_shape[0], input_shape_data[1] + pad_data[0]);
                    int hend = min(hstart + kernel_shape[1], input_shape_data[2] + pad_data[1]);
                    int wend = min(wstart + kernel_shape[2], input_shape_data[3] + pad_data[2]);
                    int pool_size = (hend - hstart) * (wend - wstart) * (zend - zstart);
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    zstart = max(zstart, 0);
                    zend = min(zend, input_shape_data[1]);
                    hend = min(hend, input_shape_data[2]);
                    wend = min(wend, input_shape_data[3]);

                    const int pool_index = (pz * output_shape_data[1] + ph) * output_shape_data[2] + pw;
                    for (int z = zstart; z < zend; ++z) {
                      for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                          const int index = (z * input_shape_data[2] + h) * input_shape_data[3] + w;
                          bottom_diff2[index] += top_diff2[pool_index] / pool_size;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          break;
        case PoolingParameter_PoolMethod_STOCHASTIC:
          NOT_IMPLEMENTED;
          break;
        default:
          LOG(FATAL) << "Unknown pooling method.";
      }
    } else {
      NOT_IMPLEMENTED;
    }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
