#ifdef USE_OCL
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void Im2colLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    im2col_gpu(bottom_data + n * this->bottom_dim_, channels_, height,
        width, kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, top_data + n * this->top_dim_);
  }
}

template <typename Dtype>
void Im2colLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  for (int n = 0; n < top[0]->num(); ++n) {
    col2im_gpu(top_diff + n * this->top_dim_, channels_, height, width,
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, bottom_diff + n * this->bottom_dim_);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(Im2colLayer);

}  // namespace caffe
#endif  // USE_OCL
