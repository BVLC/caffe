#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  global_pooling_ = pool_param.global_pooling();
  num_spatial_axes_ = bottom[0]->num_axes() - 2;
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_ = std::vector<int>(num_spatial_axes_, 0);
  if (global_pooling_) {
    CHECK(!((pool_param.kernel_size_size() > 0) ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
    for (int i = 0; i < num_spatial_axes_; ++i)
      kernel_shape_[i] = bottom[0]->shape(i + 2);
  } else {
    if (pool_param.has_kernel_h() || pool_param.has_kernel_w()) {
      CHECK_EQ(num_spatial_axes_, 2)
          << "kernel_h & kernel_w can only be used for 2D pooling.";
      CHECK_EQ(0, pool_param.kernel_size_size())
          << "Either kernel_size or kernel_h/w should be specified; not both.";
      kernel_shape_[0] = pool_param.kernel_h();
      kernel_shape_[1] = pool_param.kernel_w();
    } else {
      const int num_kernel_dims = pool_param.kernel_size_size();
      CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
          << "kernel_size must be specified once, or once per spatial "
          << "dimension (kernel_size specified " << num_kernel_dims
          << " times; " << num_spatial_axes_ << " spatial dims).";
        for (int i = 0; i < num_spatial_axes_; ++i) {
          kernel_shape_[i] =
              pool_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
        }
    }
    for (int i = 0; i < num_spatial_axes_; ++i) {
      CHECK_GT(kernel_shape_[i], 0) << "Filter dimensions must be nonzero.";
    }
  }
  // Setup stride dimensions (stride_).
  stride_ = std::vector<int>(num_spatial_axes_, 0);
  if (pool_param.has_stride_h() || pool_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D pooling.";
    CHECK_EQ(0, pool_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_[0] = pool_param.stride_h();
    stride_[1] = pool_param.stride_w();
  } else {
    const int num_stride_dims = pool_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_[i] = (num_stride_dims == 0) ? kDefaultStride :
          pool_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_ = std::vector<int>(num_spatial_axes_, 0);
  if (pool_param.has_pad_h() || pool_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D pooling.";
    CHECK_EQ(0, pool_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_[0] = pool_param.pad_h();
    pad_[1] = pool_param.pad_w();
  } else {
    const int num_pad_dims = pool_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_[i] = (num_pad_dims == 0) ? kDefaultPad :
          pool_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // remaining pooling sanity checks
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (global_pooling_) {
      CHECK(pad_[i] == 0 && stride_[i] == 1)
        << "With Global_pooling: true; only pad = 0 and stride = 1";
    }
    if (pad_[i] != 0) {
      CHECK(this->layer_param_.pooling_param().pool()
          == PoolingParameter_PoolMethod_AVE
          || this->layer_param_.pooling_param().pool()
          == PoolingParameter_PoolMethod_MAX)
          << "Padding implemented only for average and max pooling.";
    }
    CHECK_LT(pad_[i], kernel_shape_[i]);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes() - 2, num_spatial_axes_)
      << "bottom num_axes may not change.";
  channels_ = bottom[0]->shape(1);
  input_shape_ = bottom[0]->shape();
  if (global_pooling_) {
    for (int i = 0; i < num_spatial_axes_; ++i)
      kernel_shape_[i] = input_shape_[i + 2];
  }
  // setup pooled shape
  pooled_shape_ = std::vector<int>(input_shape_.size());
  pooled_shape_[0] = input_shape_[0];
  pooled_shape_[1] = input_shape_[1];
  for (unsigned int i = 0; i < num_spatial_axes_; ++i) {
    pooled_shape_[i + 2] = static_cast<int>(std::ceil(static_cast<float>(
        input_shape_[i + 2] + 2 * pad_[i] - kernel_shape_[i]) /
        stride_[i])) + 1;
  }
  for (unsigned int i = 0; i < num_spatial_axes_; ++i) {
    if (pad_[i]) {
      // If we have padding, ensure that the last pooling starts strictly
      // inside the image (instead of at the padding); otherwise clip the last.
      if ((pooled_shape_[i + 2] - 1) * stride_[i] >=
          input_shape_[i + 2] + pad_[i]) {
        --pooled_shape_[i + 2];
      }
      CHECK_LT((pooled_shape_[i + 2] - 1) * stride_[i],
          input_shape_[i + 2] + pad_[i]);
    }
  }
  // reshape outputs
  top[0]->Reshape(pooled_shape_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(pooled_shape_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(pooled_shape_);
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
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
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
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
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
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
