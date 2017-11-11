#include <vector>

#include "caffe/layers/flip_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void FlipKernel(const int num, const int channels, const int height, const int width,
                             const Dtype* src_data, Dtype* target_data, bool flip_height, bool flip_width) {
    CUDA_KERNEL_LOOP(index, num * channels * height * width) {
      int n = index / (channels * height * width);
      int cs = index % (channels * height * width);
      int c = cs / (height * width);
      int s = cs % (height * width);
      int h = s / width;
      int w = s % width;
      target_data[(((n * channels + c) * height + h) * width) + w] =
        src_data[(((n * channels + c) * height + (flip_height ? (height - 1 - h) : h)) * width) + (flip_width ? (width - 1 - w) : w)];
    }
  }

template <typename Dtype>
void FlipLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int width = bottom[0]->width();
  int height = bottom[0]->height();
  FlipKernel<Dtype> << <CAFFE_GET_BLOCKS(num * channels * height * width),
    CAFFE_CUDA_NUM_THREADS >> >(num, channels, height, width,
                                bottom_data, top_data, flip_height_, flip_width_);
}


INSTANTIATE_LAYER_GPU_FUNCS(FlipLayer);

}  // namespace caffe
