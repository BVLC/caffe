#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/saliency_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SaliencyPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SaliencyPoolingParameter saliency_pool_param = this->layer_param_.saliency_pooling_param();
  kernel_h_ = kernel_w_ = saliency_pool_param.kernel_size();
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  stride_h_ = stride_w_ = saliency_pool_param.stride();
  CHECK_GT(stride_h_, 0) << "Stride cannot be zero.";
  CHECK_GT(stride_w_, 0) << "Stride cannot be zero.";
}

template <typename Dtype>
void SaliencyPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  pooled_height_ = static_cast<int>(ceil(static_cast<float>(height_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(width_ - kernel_w_) / stride_w_)) + 1;

  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // Random number matrix init
  randoms_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  // Initialize the vector index part.
  max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);

  }

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void SaliencyPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void SaliencyPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(SaliencyPoolingLayer);
#endif

INSTANTIATE_CLASS(SaliencyPoolingLayer);

}  // namespace caffe
