
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AugmentForward(
    const int nthreads, const Dtype* bottom_data, const int num,
    const int channels, const int crop_height, const int crop_width,
    const int bottom_height, const int bottom_width, const bool mirror,
    const float* mirror_image, const float* h_shift, const float* w_shift,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % crop_width;
    int h = (index / crop_width) % crop_height;
    int c = (index / crop_width / crop_height) % channels;
    int n = index / crop_width / crop_height / channels;
    const Dtype* bottom_data_shifted =
        bottom_data + (n * channels + c) * bottom_width * bottom_height;

    int h_on = h + (int)(h_shift[n] * (bottom_height - crop_height + 1));
    int w_on = w + (int)(w_shift[n] * (bottom_width - crop_width + 1));
    int w_end = crop_width - 1 +
        (int)(w_shift[n] * (bottom_width - crop_width + 1));
    if (mirror && mirror_image[n] > .5) {
      top_data[index] = bottom_data_shifted[h_on * bottom_width + w_end - w_on];
    } else {
      top_data[index] = bottom_data_shifted[h_on * bottom_width + w_on];
    }
    if (index < 100) {
      printf("%f\n", top_data[index]);
    }
  }
}

template <typename Dtype>
void AugmentLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* input_top_data = top[0]->mutable_gpu_data();
  const Dtype* input_bottom_data = bottom[0]->gpu_data();
  const int input_top_count = top[0]->count();

  Dtype* label_top_data = top[1]->mutable_gpu_data();
  const Dtype* label_bottom_data = bottom[1]->gpu_data();
  const int label_top_count = top[1]->count();

  float* mirror_image = mirror_image_vec_.mutable_gpu_data();
  caffe_gpu_rng_uniform(num_, 0.0f, 1.0f, mirror_image);
  float* h_shift = h_shift_vec_.mutable_gpu_data();
  caffe_gpu_rng_uniform(num_, 0.0f, 1.0f, h_shift);
  float* w_shift = w_shift_vec_.mutable_gpu_data();
  caffe_gpu_rng_uniform(num_, 0.0f, 1.0f, w_shift);


  AugmentForward<Dtype><<<CAFFE_GET_BLOCKS(input_top_count),
    CAFFE_CUDA_NUM_THREADS>>>(input_top_count, input_bottom_data, num_,
        input_channels_, crop_height_, crop_width_, input_height_, input_width_,
        mirror_, mirror_image, h_shift, w_shift, input_top_data);

  AugmentForward<Dtype><<<CAFFE_GET_BLOCKS(label_top_count),
    CAFFE_CUDA_NUM_THREADS>>>(label_top_count, label_bottom_data, num_,
        label_channels_, label_crop_height_, label_crop_width_, label_height_,
        label_width_, mirror_, mirror_image, h_shift, w_shift, label_top_data);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void AugmentLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), Dtype(0),
                    bottom[i]->mutable_gpu_data());
    }
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(AugmentLayer);

}  // namespace caffe
