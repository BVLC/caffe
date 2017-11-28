#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Forward_const_gpu(bottom,top);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_const_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int bottom_dim = bottom[0]->count(this->channel_axis_);
  int top_dim = top[0]->count(this->channel_axis_);
  int num=bottom[0]->count(0, this->channel_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < num; ++n) {
      this->forward_gpu_gemm(bottom_data + n *bottom_dim , weight,
          top_data + n * top_dim);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n *top_dim, bias);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_CONST(ConvolutionLayer);

}  // namespace caffe
