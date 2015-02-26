#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaskForward(
    const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const Dtype mask_channels_count, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    Dtype true_mask_channel = bottom_data[index];
    for (Dtype m = Dtype(0); m < mask_channels_count; ++m) {
      // Obvious Form:
      // int mask_index = n * (channels * mask_channels_count * height * width) +
      //  c * (mask_channels_count * height * width) + m * (height * width) + h * width + w;
      // Compact Form:
      int mask_index = w + width * (h + height * (m + mask_channels_count * (c + channels * n)));
      if (m == true_mask_channel) {
        top_data[mask_index] = 1;
      } else {
        top_data[mask_index] = 0;
      }
    }
  }
}

template <typename Dtype>
void MaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int bottom_count = bottom[0]->count();

  MaskForward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
      bottom_count, bottom_data, num_, channels_, height_, width_,
      mask_channels_count_, top_data);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void MaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), Dtype(0),
                    bottom[i]->mutable_gpu_data());
    }
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(MaskLayer);

}  // namespace caffe
