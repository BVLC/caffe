#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/attention_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftAttentionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // bottom[0]: a (N x C x H x W)
  CHECK_EQ(bottom[0]->num_axes(), 4);
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  spatial_dim_ = bottom[0]->count(2);
  // bottom[1]: \alpha (N x 1 x H x W)
  CHECK_EQ(bottom[1]->num_axes(), 4);
  CHECK_EQ(bottom[1]->num(), bottom[0]->num());
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), bottom[0]->height());
  CHECK_EQ(bottom[1]->width(), bottom[0]->width());
  // bottom[2]: \beta (N x 1, or at least count == N)
  CHECK_EQ(bottom[2]->count(), bottom[1]->num());
  // top[0]: z (N x C x 1 x 1)
  top[0]->Reshape(num_, channels_, 1, 1);
}

template <typename Dtype>
void SoftAttentionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* a_data = bottom[0]->cpu_data();
  const Dtype* alpha_data = bottom[1]->cpu_data();
  const Dtype* beta_data = bottom[2]->cpu_data();
  Dtype* z_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < num_; ++n) {
    Dtype beta = beta_data[n];
    for (int c = 0; c < channels_; ++c) {
      z_data[n * channels_ + c] = beta * caffe_cpu_dot(spatial_dim_,
          a_data + (n * channels_ + c) * spatial_dim_,
          alpha_data + n * spatial_dim_);
    }
  }
}

template <typename Dtype>
void SoftAttentionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0] && !propagate_down[1] && !propagate_down[2]) {
    return;
  }
  const Dtype* a_data = bottom[0]->cpu_data();
  const Dtype* alpha_data = bottom[1]->cpu_data();
  const Dtype* beta_data = bottom[2]->cpu_data();
  const Dtype* z_data = top[0]->cpu_data();
  const Dtype* z_diff = top[0]->cpu_diff();
  Dtype* a_diff = bottom[0]->mutable_cpu_diff();
  Dtype* alpha_diff = bottom[1]->mutable_cpu_diff();
  Dtype* beta_diff = bottom[2]->mutable_cpu_diff();
  if (propagate_down[0]) {
    // gradient w.r.t. bottom[0]: a (N x C x H x W)
    for (int n = 0; n < num_; ++n) {
      Dtype beta = beta_data[n];
      for (int c = 0; c < channels_; ++c) {
        caffe_cpu_scale(spatial_dim_, beta * z_diff[n * channels_ + c],
            alpha_data + n * spatial_dim_,
            a_diff + (n * channels_ + c) * spatial_dim_);
      }
    }
  }
  if (propagate_down[1]) {
    // gradient w.r.t. bottom[1]: alpha (N x 1 x H x W)
    for (int n = 0; n < num_; ++n) {
      Dtype beta = beta_data[n];
      for (int s = 0; s < spatial_dim_; ++s) {
        alpha_diff[n * spatial_dim_ + s] = caffe_cpu_strided_dot(channels_,
            a_data + (n * channels_) * spatial_dim_ + s, spatial_dim_,
            z_diff + n * channels_, 1) * beta;
      }
    }
  }
  if (propagate_down[2]) {
    // gradient w.r.t. bottom[2]: beta (N x 1, or at least count == N)
    for (int n = 0; n < num_; ++n) {
      Dtype beta = beta_data[n];
      beta_diff[n] = caffe_cpu_dot(channels_, z_data + n * channels_,
          z_diff + n * channels_) / beta;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftAttentionLayer);
#endif

INSTANTIATE_CLASS(SoftAttentionLayer);
REGISTER_LAYER_CLASS(SoftAttention);

}  // namespace caffe
