#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/equiv_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void EquivPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  EquivPoolingParameter equiv_pool_param =
    this->layer_param_.equiv_pooling_param();
  CHECK(equiv_pool_param.has_kernel_size())
    << "kernel size are required.";
  if (equiv_pool_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = equiv_pool_param.kernel_size();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  pad_h_ = pad_w_ = equiv_pool_param.pad();
  stride_h_ = stride_w_ = equiv_pool_param.stride();
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK_LT(pad_h_, (kernel_h_ - 1) * stride_h_ + 1);
    CHECK_LT(pad_w_, (kernel_w_ - 1) * stride_w_ + 1);
  }
}

template <typename Dtype>
void EquivPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  equiv_pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - ((kernel_h_ - 1) * stride_h_ + 1)))) + 1;
  equiv_pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - ((kernel_w_ - 1) * stride_w_ + 1)))) + 1;
  if (pad_h_ || pad_w_) {
    // like original pooling layer, we should make sure that the last pooling
    //  starts inside the image instead of the padding region.
    if ((equiv_pooled_height_ - 1) >= height_ + pad_h_) {
      --equiv_pooled_height_;
    }
    if ((equiv_pooled_width_ - 1) >= width_ + pad_w_) {
      --equiv_pooled_width_;
    }
    CHECK_LT((equiv_pooled_height_ - 1), height_ + pad_h_);
    CHECK_LT((equiv_pooled_width_ - 1), width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, equiv_pooled_height_,
        equiv_pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // initialize the max element index vector.
  if (this->layer_param_.equiv_pooling_param().pool() ==
      EquivPoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, equiv_pooled_height_,
        equiv_pooled_width_);
  }
}

template <typename Dtype>
void EquivPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods.
  // Note that we only implement max equivalent pooling at present.
  switch (this->layer_param_.equiv_pooling_param().pool()) {
    case EquivPoolingParameter_PoolMethod_MAX:
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
          for (int ph = 0; ph < equiv_pooled_height_; ++ph) {
            for (int pw = 0; pw < equiv_pooled_width_; ++pw) {
              int hstart = ph - pad_h_;
              int wstart = pw - pad_w_;
              int hend = hstart + (stride_h_ * (kernel_h_ - 1) + 1);
              int wend = wstart + (stride_w_ * (kernel_w_ - 1) + 1);

              while (hstart < 0)
                hstart += stride_h_;
              while (wstart < 0)
                wstart += stride_w_;

              while (hend > height_)
                hend -= stride_h_;
              while (wend > width_)
                wend -= stride_w_;

              const int equiv_pool_index = ph * equiv_pooled_width_ + pw;
              for (int h = hstart; h < hend; h += stride_h_) {
                for (int w = wstart; w < wend; w += stride_w_) {
                  const int index = h * width_ + w;
                  if (bottom_data[index] > top_data[equiv_pool_index]) {
                    top_data[equiv_pool_index] = bottom_data[index];
                    if (use_top_mask) {
                      top_mask[equiv_pool_index] = static_cast<Dtype>(index);
                    } else {
                      mask[equiv_pool_index] = index;
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
    case EquivPoolingParameter_PoolMethod_AVE:
      NOT_IMPLEMENTED;
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void EquivPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  // Different pooling methods.
  // Note that we only implement max equivalent pooling at present.
  switch (this->layer_param_.equiv_pooling_param().pool()) {
    case EquivPoolingParameter_PoolMethod_MAX:
      // The main loop
      if (use_top_mask) {
        top_mask = top[1]->cpu_data();
      } else {
        mask = max_idx_.cpu_data();
      }
      for (int n = 0; n < top[0]->num(); ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int ph = 0; ph < equiv_pooled_height_; ++ph) {
            for (int pw = 0; pw < equiv_pooled_width_; ++pw) {
              const int index = ph * equiv_pooled_width_ + pw;
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
    case EquivPoolingParameter_PoolMethod_AVE:
      NOT_IMPLEMENTED;
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(EquivPoolingLayer);
#endif

INSTANTIATE_CLASS(EquivPoolingLayer);
REGISTER_LAYER_CLASS(EquivPooling);
}  // namespace caffe
