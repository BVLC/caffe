#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void CopyForward(const int nthreads, const Dtype* bottom_a,
                            const Dtype* bottom_b, Dtype* top, int num,
                            int channels_a, int channels_b, int height_a,
                            int width_a, int height_b, int width_b) {

  CUDA_KERNEL_LOOP(index, nthreads)
  {

    int pad_h = (height_b - height_a) / 2;
    int pad_w = (width_b - width_a) / 2;

    int batch_id = index / (channels_a * channels_b * height_a * width_a);

    int bottom_id = ((index
        - batch_id * channels_a * channels_b * height_a * width_a)
        / (channels_a * height_a * width_a)) % 2;

    int h = ((index / width_a) % height_a);
    int w = (index % width_a);

    if (bottom_id == 0) {
      int channel_id = (index / ((width_a * height_a)) % channels_a);
      int aidx = ((((batch_id) * channels_a + channel_id) * height_a + h)
          * width_a + w);
      top[index] = bottom_a[aidx];
    } else {
      int channel_id = (index / ((width_a * height_a)) % channels_b);
      int bidx =
          ((((batch_id) * channels_b + channel_id) * height_a + h + pad_h)
              * width_a + w + (h * 2 + 1) * pad_w);
      top[index] = bottom_b[bidx];
    }
  }

}

template<typename Dtype>
__global__ void CopyBackward(const int nthreads, Dtype* bottom_a,
                             const Dtype* top, int num, int channels_a,
                             int channels_b, int height_a, int width_a,
                             int height_b, int width_b) {

  CUDA_KERNEL_LOOP(index, nthreads)
  {

    int batch_id = index / (channels_a * channels_b * height_a * width_a);

    int bottom_id = ((index
        - batch_id * channels_a * channels_b * height_a * width_a)
        / (channels_a * height_a * width_a)) % 2;

    int h = ((index / width_a) % height_a);
    int w = (index % width_a);

    if (bottom_id == 0) {
      int channel_id = (index / ((width_a * height_a)) % channels_a);
      int aidx = ((((batch_id) * channels_a + channel_id) * height_a + h)
          * width_a + w);
      bottom_a[aidx] = top[index];
    }
  }

}
#endif // USE_CUDA

template<typename Dtype>
void MergeCropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {

  int count = top[0]->count() * 2;

  const Dtype* bottom_data_a = bottom[0]->gpu_data();
  const Dtype* bottom_data_b = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  int num = bottom[0]->num();

  // All channels of both inputs are copied
  int channels_a = bottom[0]->channels();
  int channels_b = bottom[1]->channels();

  // Width and height of the smaller input, which should be input 0
  int height_a = bottom[0]->height();
  int width_a = bottom[0]->width();

  int height_b = bottom[1]->height();
  int width_b = bottom[1]->width();

  CopyForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS) (
      count, bottom_data_a, bottom_data_b, top_data, num, channels_a,
      channels_b, height_a, width_a, height_b, width_b);

}

template<typename Dtype>
void MergeCropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  int count = top[0]->count() * 2;

  Dtype* bottom_diff_a = bottom[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();

  int num = bottom[0]->num();

  // All channels of both inputs are copied
  int channels_a = bottom[0]->channels();
  int channels_b = bottom[1]->channels();

  // Width and height of the smaller input, which should be input 0
  int height_a = bottom[0]->height();
  int width_a = bottom[0]->width();

  int height_b = bottom[1]->height();
  int width_b = bottom[1]->width();

  CopyBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS) (
      count, bottom_diff_a, top_diff, num, channels_a, channels_b, height_a,
      width_a, height_b, width_b);
}

INSTANTIATE_LAYER_GPU_FUNCS(MergeCropLayer);

}  // namespace caffe
