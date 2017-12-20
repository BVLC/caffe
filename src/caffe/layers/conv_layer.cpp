#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
vector<int> ConvolutionLayer<Dtype>::compute_output_shape() const {
  const int *kernel_shape_data = this->kernel_shape_.cpu_data();
  const int *stride_data = this->stride_.cpu_data();
  const int *pad_data = this->pad_.cpu_data();
  const int *dilation_data = this->dilation_.cpu_data();
  vector<int> output_shape;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(this->channel_axis_, i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim =
        (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
    output_shape.push_back(output_dim);
  }
  return output_shape;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  Forward_const_cpu(bottom, top);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_const_cpu(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  const Dtype *weight = this->blobs_[0]->cpu_data();
  int bottom_dim = bottom[0]->count(this->channel_axis_);
  int top_dim = top[0]->count(this->channel_axis_);
  int num = bottom[0]->count(0, this->channel_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype *bottom_data = bottom[i]->cpu_data();
    Dtype *top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < num; ++n) {
      this->forward_cpu_gemm(bottom_data + n * bottom_dim, weight,
                             top_data + n * top_dim);
      if (this->bias_term_) {
        const Dtype *bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * top_dim, bias);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
STUB_GPU_FORWARD_CONST(ConvolutionLayer, Forward_const);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

} // namespace caffe
