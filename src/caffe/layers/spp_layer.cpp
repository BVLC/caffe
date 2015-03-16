#include <algorithm>
#include <cmath>
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
void SPPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SPPParameter spp_param = this->layer_param_.spp_param();
  CHECK(spp_param.has_kernel_depth())
      << "Needs kernel depth.";
  kernel_depth_ = spp_param.kernel_depth();
  CHECK_GT(kernel_depth_, 0) << "Kernel depth must be greater than zero.";
  if (spp_param.has_image_w()) {
    image_w_ = spp_param.image_w();
    image_h_ = spp_param.image_h();
  }
}

template <typename Dtype>
void SPPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_GT(height_, std::pow(2, (kernel_depth_ - 1))) << "SPP Kernel too deep.";
  CHECK_GT(width_, std::pow(2, (kernel_depth_ - 1))) << "SPP Kernel too deep.";

  // sum_i=1^n i^2 = i(i+1)(2i+1)/6
  output_size_ = kernel_depth_ * (kernel_depth_ + 1) * (2 * kernel_depth_ + 1) / 6;

  // For windowed case.
  if (bottom.size() > 1) {
    top[0]->Reshape(bottom[0]->num(), channels_, output_size_, bottom[1]->height());
  } else {
    top[0]->Reshape(bottom[0]->num(), channels_, output_size_, 1);
  }
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // Holds the index where the max was found for backprop.
  else if (top.size() == 1) {
    if (bottom.size() > 1) {
      max_idx_.Reshape(bottom[0]->num(), channels_, output_size_, bottom[1]->height());
    } else {
      max_idx_.Reshape(bottom[0]->num(), channels_, output_size_, 1);
    }
  }
}

template <typename Dtype>
void SPPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  // Checks for the windowed case where many outputs are required.
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  if (bottom.size() > 1) {
    // Windowed case.
    const int window_count = bottom[1]->height();
    const Dtype* bottom_window_data = bottom[1]->cpu_data();
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
        for (int win = 0; win < window_count; ++win) {
          int pool_index = 0;
          // 4 = number of coordinates per row.
          const Dtype* bottom_window_data_shifted = bottom_window_data + (n * window_count + win) * 4;
          float window_x = bottom_window_data_shifted[0] * width_ / image_w_;
          float window_y = bottom_window_data_shifted[1] * height_ / image_h_;
          // Make sure window is over at least a 1x1 patch.
          float window_w = max((float)bottom_window_data_shifted[2] * width_ / image_w_, 1.0f);
          float window_h = max((float)bottom_window_data_shifted[3] * height_ / image_h_, 1.0f);
          for (int num_pools = 1; num_pools < kernel_depth_; ++num_pools) {
            // Using fractional heights to better represent smaller sections
            // instead of defaulting to repeating the end pixels over and over.
            float kernel_h = window_h  / num_pools;
            float kernel_w = window_w / num_pools;
            for (int ph = 0; ph < num_pools; ++ph) {
              for (int pw = 0; pw < num_pools; ++pw) {
                float hstart_float = ph * kernel_h + window_y;
                float wstart_float = pw * kernel_w + window_x;
                // Starting point is floor of possible window.
                int hstart = (int)hstart_float;
                int wstart = (int)wstart_float;
                // Ending point is ceiling of possible window bounded by
                // dimensions of image and dimensions of patch.
                int hend = min(min((int)ceil(hstart_float + kernel_h), height_), (int)ceil(window_y + window_h));
                int wend = min(min((int)ceil(wstart_float + kernel_w), width_), (int)ceil(window_x + window_w));
                // Make sure over at least 1x1 patch for sure. Could have happened if hit
                // image size constraint I think.
                if (hstart == hend) {
                  if (hend < height_) {
                    ++hend;
                  } else {
                    --hstart;
                  }
                }
                if (wstart == wend) {
                  if (wend < width_) {
                    ++wend;
                  } else {
                    --wstart;
                  }
                }
                // Initialize max to first element.
                top_data[pool_index] = bottom_data[hstart * width_ + wstart];
                if (use_top_mask) {
                  top_mask[pool_index] = static_cast<Dtype>(hstart * width_ + wstart);
                } else {
                  mask[pool_index] = hstart * width_ + wstart;
                }
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
                ++pool_index;
              }
            }
          }
        }
        // compute offset
        // std::cout << "Bottom offset: " << bottom[0]->offset(0, 1);
        // std::cout << "Top offset: " << top[0]->offset(0, 1);
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
  } else {
    // Single window case.
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
        int pool_index = 0;
        for (int num_pools = 0; num_pools < kernel_depth_; ++num_pools) {
          // Trick to get the kernel over the whole zone.
          int kernel_h = (height_ + num_pools - 1) / num_pools;
          int kernel_w = (width_ + num_pools - 1) / num_pools;
          for (int ph = 0; ph < num_pools; ++ph) {
            for (int pw = 0; pw < num_pools; ++pw) {
              int hstart = ph * kernel_h;
              int wstart = pw * kernel_w;
              int hend = min(hstart + kernel_h, height_);
              int wend = min(wstart + kernel_w, width_);
              // Not sure how their initialization works, so I'll do my own.
              top_data[pool_index] = bottom_data[hstart * width_ + wstart];
              if (use_top_mask) {
                top_mask[pool_index] = static_cast<Dtype>(hstart * width_ + wstart);
              } else {
                mask[pool_index] = hstart * width_ + wstart;
              }
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
              ++pool_index;
            }
          }
        }
        // compute offset
        // std::cout << "Bottom offset: " << bottom[0]->offset(0, 1);
        // std::cout << "Top offset: " << top[0]->offset(0, 1);
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
  }
}

template <typename Dtype>
void SPPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
  // The main loop
  if (use_top_mask) {
    top_mask = top[1]->cpu_data();
  } else {
    mask = max_idx_.cpu_data();
  }
  if (bottom.size() > 1) {
    const int window_count = bottom[1]->height();
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int win = 0; win < window_count; ++win) {
          for (int i = 0; i < output_size_; ++i) {
            const int bottom_index =
                use_top_mask ? top_mask[i] : mask[i];
            bottom_diff[bottom_index] += top_diff[i];
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
    }
  }
  else{
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int i = 0; i < output_size_; ++i) {
          const int bottom_index =
              use_top_mask ? top_mask[i] : mask[i];
          bottom_diff[bottom_index] += top_diff[i];
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
  }
}


#ifdef CPU_ONLY
STUB_GPU(SPPLayer);
#endif

INSTANTIATE_CLASS(SPPLayer);
REGISTER_LAYER_CLASS(SPP);

}  // namespace caffe
