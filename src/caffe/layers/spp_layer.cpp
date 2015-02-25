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
      vector<Blob<Dtype>*>* top) {
  SPPParameter spp_param = this->layer_param_.spp_param();
  CHECK(spp_param.has_kernel_depth())
      << "Needs kernel depth.";
  kernel_depth_ = spp_param.kernel_depth();
  CHECK_GT(kernel_depth_, 0) << "Kernel depth cannot be zero.";
}

template <typename Dtype>
void SPPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_GT(height_, std::pow(2, (kernel_depth_ - 1))) << "SPP Kernel too deep.";
  CHECK_GT(width_, std::pow(2, (kernel_depth_ - 1))) << "SPP Kernel too deep.";

  output_size_ = 0;
  for (int i = 0; i < kernel_depth_; ++i) {
    output_size_ += (1 << i)* (1 << i) ;
  }
  (*top)[0]->Reshape(bottom[0]->num(), channels_, output_size_, 1);
  if (top->size() > 1) {
    (*top)[1]->ReshapeLike(*(*top)[0]);
  }
  // Holds the index where the max was found for backprop.
  else if (top->size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, output_size_, 1);
  }
}

template <typename Dtype>
void SPPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int top_count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Initialize
  if (use_top_mask) {
    top_mask = (*top)[1]->mutable_cpu_data();
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
      for (int d = 0; d < kernel_depth_; ++d) {
        int num_pools = std::pow(2, d);
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
            top_data[pool_index] = bottom_data[0];
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
      bottom_data += bottom[0]->offset(0, 1);
      top_data += (*top)[0]->offset(0, 1);
      if (use_top_mask) {
        top_mask += (*top)[0]->offset(0, 1);
      } else {
        mask += (*top)[0]->offset(0, 1);
      }
    }
  }
}

template <typename Dtype>
void SPPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
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
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int i = 0; i < output_size_; ++i) { 
        const int bottom_index =
            use_top_mask ? top_mask[i] : mask[i];
        bottom_diff[bottom_index] += top_diff[i];
      }
      bottom_diff += (*bottom)[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
      if (use_top_mask) {
        top_mask += top[0]->offset(0, 1);
      } else {
        mask += top[0]->offset(0, 1);
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SPPLayer);
#endif

INSTANTIATE_CLASS(SPPLayer);


}  // namespace caffe
