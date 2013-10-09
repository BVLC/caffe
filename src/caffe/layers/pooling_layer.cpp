// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#define CAFFE_MAX_POOLING_THRESHOLD 1e-8f

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void PoolingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "PoolingLayer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "PoolingLayer takes a single blob as output.";
  KSIZE_ = this->layer_param_.kernelsize();
  STRIDE_ = this->layer_param_.stride();
  CHANNELS_ = bottom[0]->channels();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();
  POOLED_HEIGHT_ = static_cast<int>(
      ceil(static_cast<float>(HEIGHT_ - KSIZE_) / STRIDE_)) + 1;
  POOLED_WIDTH_ = static_cast<int>(
      ceil(static_cast<float>(WIDTH_ - KSIZE_) / STRIDE_)) + 1;
  (*top)[0]->Reshape(bottom[0]->num(), CHANNELS_, POOLED_HEIGHT_,
      POOLED_WIDTH_);
};

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  int top_count = (*top)[0]->count();
  switch (this->layer_param_.pool()) {
  case LayerParameter_PoolMethod_MAX:
    // Initialize
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = -FLT_MAX;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < CHANNELS_; ++c) {
        for (int ph = 0; ph < POOLED_HEIGHT_; ++ph) {
          for (int pw = 0; pw < POOLED_WIDTH_; ++pw) {
            int hstart = ph * STRIDE_;
            int wstart = pw * STRIDE_;
            int hend = min(hstart + KSIZE_, HEIGHT_);
            int wend = min(wstart + KSIZE_, WIDTH_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * POOLED_WIDTH_ + pw] =
                  max(top_data[ph * POOLED_WIDTH_ + pw],
                      bottom_data[h * WIDTH_ + w]);
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
      }
    }
    break;
  case LayerParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < CHANNELS_; ++c) {
        for (int ph = 0; ph < POOLED_HEIGHT_; ++ph) {
          for (int pw = 0; pw < POOLED_WIDTH_; ++pw) {
            int hstart = ph * STRIDE_;
            int wstart = pw * STRIDE_;
            int hend = min(hstart + KSIZE_, HEIGHT_);
            int wend = min(wstart + KSIZE_, WIDTH_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * POOLED_WIDTH_ + pw] +=
                    bottom_data[h * WIDTH_ + w];
              }
            }
            top_data[ph * POOLED_WIDTH_ + pw] /=
                (hend - hstart) * (wend - wstart);
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
Dtype PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) {
    return Dtype(0.);
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  memset(bottom_diff, 0, (*bottom)[0]->count() * sizeof(Dtype));
  switch (this->layer_param_.pool()) {
  case LayerParameter_PoolMethod_MAX:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < CHANNELS_; ++c) {
        for (int ph = 0; ph < POOLED_HEIGHT_; ++ph) {
          for (int pw = 0; pw < POOLED_WIDTH_; ++pw) {
            int hstart = ph * STRIDE_;
            int wstart = pw * STRIDE_;
            int hend = min(hstart + KSIZE_, HEIGHT_);
            int wend = min(wstart + KSIZE_, WIDTH_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * WIDTH_ + w] +=
                    top_diff[ph * POOLED_WIDTH_ + pw] *
                    (bottom_data[h * WIDTH_ + w] >=
                        top_data[ph * POOLED_WIDTH_ + pw] -
                        CAFFE_MAX_POOLING_THRESHOLD);
              }
            }
          }
        }
        // offset
        bottom_data += (*bottom)[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case LayerParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < CHANNELS_; ++c) {
        for (int ph = 0; ph < POOLED_HEIGHT_; ++ph) {
          for (int pw = 0; pw < POOLED_WIDTH_; ++pw) {
            int hstart = ph * STRIDE_;
            int wstart = pw * STRIDE_;
            int hend = min(hstart + KSIZE_, HEIGHT_);
            int wend = min(wstart + KSIZE_, WIDTH_);
            int poolsize = (hend - hstart) * (wend - wstart);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * WIDTH_ + w] +=
                  top_diff[ph * POOLED_WIDTH_ + pw] / poolsize;
              }
            }
          }
        }
        // offset
        bottom_data += (*bottom)[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  return Dtype(0.);
}


INSTANTIATE_CLASS(PoolingLayer);


}  // namespace caffe
