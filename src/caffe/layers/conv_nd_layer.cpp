#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#if defined(USE_GREENTEA) && !defined(USE_CUDA)
#include "conv_nd_layer.cu"
#endif

namespace caffe {

template <typename Dtype>
void ConvolutionNDLayer<Dtype>::compute_output_shape() {
  // input_shape_ + 1 to skip channel axis
  const int* input_shape_data = this->input_shape_.cpu_data() + 1;
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    const int input_dim = input_shape_data[i];
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i])
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionNDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void ConvolutionNDLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionNDLayer);
#endif

INSTANTIATE_CLASS(ConvolutionNDLayer);

}  // namespace caffe
