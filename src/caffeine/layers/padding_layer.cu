#include "caffeine/layer.hpp"
#include "caffeine/vision_layers.hpp"

#include <iostream>

namespace caffeine {

template <typename Dtype>
void PaddingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  PAD_ = this->layer_param_.pad();
  CHECK_EQ(bottom.size(), 1) << "Padding Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Padding Layer takes a single blob as output.";
  NUM_ = bottom[0]->num();
  CHANNEL_ = bottom[0]->channels();
  HEIGHT_IN_ = bottom[0]->height();
  WIDTH_IN_ = bottom[0]->width();
  HEIGHT_OUT_ = HEIGHT_IN_ + PAD_ * 2;
  WIDTH_OUT_ = WIDTH_IN_ + PAD_ * 2;
  (*top)[0]->Reshape(NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_);

};

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  memset(top_data, 0, sizeof(Dtype) * (*top)[0]->count());
  // In short, top[n, c, h, w] = bottom[n, c, h-pad, w-pad] if in range
  for (int n = 0; n < NUM_; ++n) {
    for (int c = 0; c < CHANNEL_; ++c) {
      for (int h = 0; h < HEIGHT_IN_; ++h) {
        // copy the width part
        memcpy(
            top_data + ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h + PAD_)
                * WIDTH_OUT_ + PAD_,
            bottom_data + ((n * CHANNEL_ + c) * HEIGHT_IN_ + h) * WIDTH_IN_,
            sizeof(Dtype) * WIDTH_IN_);
      }
    }
  }
}

template <typename Dtype>
Dtype PaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  //memset(bottom_data, 0, sizeof(Dtype) * (*bottom)[0]->count());
  for (int n = 0; n < NUM_; ++n) {
    for (int c = 0; c < CHANNEL_; ++c) {
      for (int h = 0; h < HEIGHT_IN_; ++h) {
        // copy the width part
        memcpy(
            bottom_diff + ((n * CHANNEL_ + c) * HEIGHT_IN_ + h) * WIDTH_IN_,
            top_diff + ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h + PAD_)
                * WIDTH_OUT_ + PAD_,
            sizeof(Dtype) * WIDTH_IN_);
      }
    }
  }
  return Dtype(0.);
}

template <typename Dtype>
__global__ void PaddingForward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int pad) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) {
    int height_out = height_in + pad + pad;
    int width_out = width_in + pad + pad;
    int w = index % width_in;
    index /= width_in;
    int h = index % height_in;
    index /= height_in;
    int c = index % channel;
    index /= channel;
    out[((index * channel + c) * height_out + h + pad) * width_out + pad + w] =
        in[((index * channel + c) * height_in + h) * width_in + w];
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // First, set all data to be zero for the boundary pixels
  CUDA_CHECK(cudaMemset(top_data, 0, sizeof(Dtype) * (*top)[0]->count()));
  PaddingForward<Dtype><<<CAFFEINE_GET_BLOCKS(count), CAFFEINE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, NUM_, CHANNEL_, HEIGHT_IN_, WIDTH_IN_,
      PAD_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void PaddingBackward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int pad) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) {
    int height_out = height_in + pad + pad;
    int width_out = width_in + pad + pad;
    int w = index % width_in;
    index /= width_in;
    int h = index % height_in;
    index /= height_in;
    int c = index % channel;
    index /= channel;
    out[((index * channel + c) * height_in + h) * width_in + w] =
        in[((index * channel + c) * height_out + h + pad) * width_out + pad + w];
  }
}

template <typename Dtype>
Dtype PaddingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    PaddingBackward<Dtype><<<CAFFEINE_GET_BLOCKS(count), CAFFEINE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_diff, NUM_, CHANNEL_, HEIGHT_IN_, WIDTH_IN_,
        PAD_);
    CUDA_POST_KERNEL_CHECK;
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(PaddingLayer);


}  // namespace caffeine
