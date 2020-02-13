#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void UnpoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const PoolingParameter_PoolMethod pool =
      this->layer_param_.pooling_param().pool();
  // The last input blob gives the output shape.
  kernel_size_ = this->layer_param_.pooling_param().kernel_size();
  stride_ = this->layer_param_.pooling_param().stride();
  pad_ = this->layer_param_.pooling_param().pad();
  channels_ = bottom[0]->channels();
  pooled_height_ = bottom[0]->height();
  pooled_width_ = bottom[0]->width();
  if (bottom.size() >= 2 || pool == PoolingParameter_PoolMethod_MAX) {
    const int shape_blob_index = bottom.size() - 1;
    height_ = bottom[shape_blob_index]->height();
    width_ = bottom[shape_blob_index]->width();
    int expected_pooled_height = static_cast<int>(ceil(static_cast<float>(
        height_ + 2 * pad_ - kernel_size_) / stride_)) + 1;
    int expected_pooled_width = static_cast<int>(ceil(static_cast<float>(
        width_ + 2 * pad_ - kernel_size_) / stride_)) + 1;
    if (pad_) {
      // If we have padding, ensure that the last convolution starts strictly
      // inside the image (instead of at the padding); otherwise clip the last.
      if ((expected_pooled_height - 1) * stride_ >= height_ + pad_) {
        --expected_pooled_height;
      }
      if ((expected_pooled_width - 1) * stride_ >= width_ + pad_) {
        --expected_pooled_width;
      }
      CHECK_LT((expected_pooled_height - 1) * stride_, height_ + pad_);
      CHECK_LT((expected_pooled_width - 1) * stride_, width_ + pad_);
    }
    CHECK_EQ(expected_pooled_height, pooled_height_);
    CHECK_EQ(expected_pooled_width, pooled_width_);
    CHECK_EQ(bottom[0]->num(), bottom[shape_blob_index]->num());
    CHECK_EQ(channels_, bottom[shape_blob_index]->channels());
  } else {
    height_= stride_ * (pooled_height_ - 1) + kernel_size_ - 2 * pad_;
    width_ = stride_ * (pooled_width_ - 1) + kernel_size_ - 2 * pad_;
  }
  top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(channels_, bottom[1]->channels());
  if (pool == PoolingParameter_PoolMethod_MAX) {
    CHECK_EQ(pooled_height_, bottom[1]->height());
    CHECK_EQ(pooled_width_, bottom[1]->width());
  }
  if (pool == PoolingParameter_PoolMethod_AVE) {
    pool_count_.Reshape(1, 1, height_, width_);
    Dtype* pool_count = pool_count_.mutable_cpu_data();
    caffe_set(pool_count_.count(), Dtype(0), pool_count);
    for (int ph = 0; ph < pooled_height_; ++ph) {
      for (int pw = 0; pw < pooled_width_; ++pw) {
        int hstart = ph * stride_ - pad_;
        int wstart = pw * stride_ - pad_;
        int hend = min(hstart + kernel_size_, height_ + pad_);
        int wend = min(wstart + kernel_size_, width_ + pad_);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height_);
        wend = min(wend, width_);
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            ++pool_count[h * width_ + w];
          }
        }
      }
    }
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const bool overlapping = (stride_ < kernel_size_);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_data);
  const Dtype* mask;
  const Dtype* pool_count;
  const Dtype* argmax_count = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize to 0s.
    mask = bottom[1]->cpu_data();
    if (overlapping) {
      argmax_count = bottom[2]->cpu_data();
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int top_index = static_cast<int>(mask[index]);
            if (argmax_count) {
              top_data[top_index] +=
                  bottom_data[index] / argmax_count[top_index];
            } else {
              top_data[top_index] += bottom_data[index];
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        mask += bottom[1]->offset(0, 1);
        if (overlapping) {
          argmax_count += bottom[2]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    pool_count = pool_count_.cpu_data();
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_ - pad_;
            int wstart = pw * stride_ - pad_;
            int hend = min(hstart + kernel_size_, height_);
            int wend = min(wstart + kernel_size_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[h * width_ + w] +=
                    bottom_data[ph * pooled_width_ + pw] /
                    pool_count[h * width_ + w];
              }
            }
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
void UnpoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const bool overlapping = (stride_ < kernel_size_);
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* argmax_count = NULL;
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  const Dtype* mask;
  const Dtype* pool_count;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    mask = bottom[1]->cpu_data();
    if (overlapping) {
      argmax_count = bottom[2]->cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int pool_index = ph * pooled_width_ + pw;
            const int top_index = static_cast<int>(mask[pool_index]);
            bottom_diff[pool_index] += top_diff[top_index];
          }
        }
        if (overlapping) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              const int pool_index = ph * pooled_width_ + pw;
              const int top_index = static_cast<int>(mask[pool_index]);
              bottom_diff[pool_index] /= argmax_count[top_index];
            }
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        mask += bottom[1]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (overlapping) {
          argmax_count += bottom[2]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    pool_count = pool_count_.cpu_data();
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_ - pad_;
            int wstart = pw * stride_ - pad_;
            int hend = min(hstart + kernel_size_, height_ + pad_);
            int wend = min(wstart + kernel_size_, width_ + pad_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[ph * pooled_width_ + pw] +=
                    top_diff[h * width_ + w] / pool_count[h * width_ + w];
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
STUB_GPU(UnpoolingLayer);
#endif

INSTANTIATE_CLASS(UnpoolingLayer);
REGISTER_LAYER_CLASS(Unpooling);

}  // namespace caffe
