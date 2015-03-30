#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.do_spm()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With spm pooling: filter size cannot specified";
    CHECK(!pool_param.has_pool_size() !=
      !(pool_param.has_pool_h() && pool_param.has_pool_w()))
      << "SPM pool size is pool_size OR pool_h and pool_w; not both";
    CHECK(pool_param.has_pool_size() ||
      (pool_param.has_pool_h() && pool_param.has_pool_w()))
      << "For non-square SPM output size both pool_h and pool_w are required.";
  } else {
    CHECK(!(pool_param.has_pool_size() ||
      pool_param.has_pool_h() || pool_param.has_pool_w()))
      << "With non-spm pooling: pool size cannot specified";
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  do_spm_ = pool_param.do_spm();
  if (do_spm_) {
    if (pool_param.has_pool_size()) {
      pool_h_ = pool_w_ = pool_param.pool_size();
    } else {
      pool_h_ = pool_param.pool_h();
      pool_w_ = pool_param.pool_w();
    }
    CHECK_GT(pool_h_, 0) << "Pool size cannot be zero.";
    CHECK_GT(pool_w_, 0) << "Pool size cannot be zero.";
    CHECK(pad_h_ == 0 && pad_w_ == 0)
      << "With SPM pooling: true, only allow pad = 0";
  }
  else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
    CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(pool_param.pool()
        == PoolingParameter_PoolMethod_AVE
        || pool_param.pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
  }
  shrink_factor_ = pool_param.shrink_factor();
  CHECK_GE(shrink_factor_, 1);
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (!do_spm_) {
    pool_h_ = static_cast<int>(ceil(static_cast<float>(
       height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    pool_w_ = static_cast<int>(ceil(static_cast<float>(
       width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    if (pad_h_ || pad_w_) {
      // If we have padding, ensure that the last pooling starts strictly
      // inside the image (instead of at the padding); otherwise clip the last.
      if ((pool_h_ - 1) * stride_h_ >= height_ + pad_h_) {
	--pool_h_;
      }
      if ((pool_w_ - 1) * stride_w_ >= width_ + pad_w_) {
	--pool_w_;
      }
      CHECK_LT((pool_h_ - 1) * stride_h_, height_ + pad_h_);
      CHECK_LT((pool_w_ - 1) * stride_w_, width_ + pad_w_);
    }
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pool_h_, pool_w_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pool_h_, pool_w_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pool_h_, pool_w_);
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
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();
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
      if (do_spm_) {
	height_ = (bottom[1]->cpu_data(n)[0] - 1) / shrink_factor_ + 1;
	width_  = (bottom[1]->cpu_data(n)[1] - 1) / shrink_factor_ + 1;
	CHECK_LE(height_, bottom_height);
	CHECK_LE(width_, bottom_width);
	kernel_h_ = height_ / pool_h_;
	kernel_w_ = width_ / pool_w_;
	stride_h_ = kernel_h_;
	stride_w_ = kernel_w_;
      }
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pool_h_; ++ph) {
          for (int pw = 0; pw < pool_w_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pool_w_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * bottom_width + w;
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
      if (do_spm_) {
	height_ = (bottom[1]->cpu_data(n)[0] - 1) / shrink_factor_ + 1;
	width_  = (bottom[1]->cpu_data(n)[1] - 1) / shrink_factor_ + 1;
	CHECK_LE(height_, bottom_height);
	CHECK_LE(width_, bottom_width);
	kernel_h_ = height_ / pool_h_;
	kernel_w_ = width_ / pool_w_;
	stride_h_ = kernel_h_;
	stride_w_ = kernel_w_;
      }
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pool_h_; ++ph) {
          for (int pw = 0; pw < pool_w_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_area = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pool_w_ + pw] +=
                    bottom_data[h * bottom_width + w];
              }
            }
            top_data[ph * pool_w_ + pw] /= pool_area;
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
  if (do_spm_ && propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to size inputs.";
  }
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int bottom_width = bottom[0]->width();
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
      if (do_spm_) {
	height_ = (bottom[1]->cpu_data(n)[0] - 1) / shrink_factor_ + 1;
	width_  = (bottom[1]->cpu_data(n)[1] - 1) / shrink_factor_ + 1;
	kernel_h_ = height_ / pool_h_;
	kernel_w_ = width_ / pool_w_;
	stride_h_ = kernel_h_;
	stride_w_ = kernel_w_;
      }
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pool_h_; ++ph) {
          for (int pw = 0; pw < pool_w_; ++pw) {
            const int index = ph * pool_w_ + pw;
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
      if (do_spm_) {
	height_ = (bottom[1]->cpu_data(n)[0] - 1) / shrink_factor_ + 1;
	width_  = (bottom[1]->cpu_data(n)[1] - 1) / shrink_factor_ + 1;
	kernel_h_ = height_ / pool_h_;
	kernel_w_ = width_ / pool_w_;
	stride_h_ = kernel_h_;
	stride_w_ = kernel_w_;
      }
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pool_h_; ++ph) {
          for (int pw = 0; pw < pool_w_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_area = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * bottom_width + w] +=
                  top_diff[ph * pool_w_ + pw] / pool_area;
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
