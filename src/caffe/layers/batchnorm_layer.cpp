#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BatchnormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  num_ = bottom[0]->num();
  bottom_size_ = bottom[0]->count() / bottom[0]->num();

  // Initialize the beta and gamma blobs to 1
  this->blobs_.resize(2);
  for (int i = 0; i < 2; ++i) {
    this->blobs_[i].reset(new Blob<Dtype>(
      1, bottom_size_, 1, 1));
    caffe_set(this->blobs_[i]->count(), i == 0 ? Dtype(1) : Dtype(0),
      this->blobs_[i]->mutable_cpu_data());
  }

  batch_mean_.Reshape(1, bottom_size_, 1, 1);
  buffer_blob_.Reshape(1, bottom_size_, 1, 1);
  batch_variance_.Reshape(1, bottom_size_, 1, 1);
  var_epsilon_ = Dtype(0.1);

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BatchnormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* gamma_data = this->blobs_[0]->cpu_data();
  const Dtype* beta_data = this->blobs_[1]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  Dtype* mean_data = batch_mean_.mutable_cpu_data();
  Dtype* variance_data = batch_variance_.mutable_cpu_data();
  Dtype* buffer = buffer_blob_.mutable_cpu_data();

  caffe_set(bottom_size_, Dtype(0), mean_data);
  caffe_set(bottom_size_, Dtype(0), variance_data);

  for (int n = 0; n < num_; ++n) {
    caffe_add(bottom_size_, bottom_data + bottom[0]->offset(n), mean_data,
        mean_data);
    caffe_sqr(bottom_size_, bottom_data + bottom[0]->offset(n), buffer);
    caffe_add(bottom_size_, buffer, variance_data, variance_data);
  }
  caffe_cpu_scale(bottom_size_, Dtype(1) / Dtype(num_), mean_data, mean_data);
  caffe_cpu_scale(bottom_size_, Dtype(1) / Dtype(num_), variance_data,
      variance_data);

  caffe_sqr(bottom_size_, mean_data, buffer);
  caffe_sub(bottom_size_, variance_data, buffer, variance_data);
  caffe_add_scalar(bottom_size_, var_epsilon_, variance_data);
  caffe_powx(bottom_size_, variance_data, Dtype(0.5), variance_data);

  for (int n = 0; n < num_; ++n) {
    caffe_sub(bottom_size_, bottom_data + bottom[0]->offset(n), mean_data, buffer);
    caffe_div(bottom_size_, buffer, variance_data, buffer);
    caffe_mul(bottom_size_, buffer, gamma_data, buffer);
    caffe_add(bottom_size_, buffer, beta_data,
        top_data + (*top)[0]->offset(n));
  }
}

template <typename Dtype>
void BatchnormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* variance_data = batch_variance_.cpu_data();
  const Dtype* gamma_data = this->blobs_[0]->cpu_data();
  const Dtype* beta_data = this->blobs_[1]->cpu_data();

  Dtype* dl_dvar = batch_variance_.mutable_cpu_diff();
  Dtype* dl_dmean = batch_mean_.mutable_cpu_diff();
  Dtype* buffer = buffer_blob_.mutable_cpu_data();

  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* gamma_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* beta_diff = this->blobs_[1]->mutable_cpu_diff();

  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
  caffe_set(this->blobs_[0]->count(), Dtype(0), gamma_diff);
  caffe_set(this->blobs_[1]->count(), Dtype(0), beta_diff);
  caffe_set(bottom_size_, Dtype(0), dl_dvar);
  caffe_set(bottom_size_, Dtype(0), dl_dmean);

  for (int n = 0; n < num_; ++n) {
    // fill gamma_diff
    caffe_sub(bottom_size_, top_data + top[0]->offset(n), beta_data,
        buffer);
    caffe_div(bottom_size_, buffer, gamma_data,
        buffer);
    caffe_mul(bottom_size_, buffer, top_diff + top[0]->offset(n),
        buffer);
    caffe_add(bottom_size_, buffer, gamma_diff, gamma_diff);

    // fill beta_diff
    caffe_add(bottom_size_, top_diff + top[0]->offset(n), beta_diff, beta_diff);
  }

  // fill bottom_diff direct term
  for (int n = 0; n < num_; ++n) {
    caffe_mul(bottom_size_, top_diff + top[0]->offset(n), gamma_data, buffer);
    caffe_div(bottom_size_, buffer, variance_data, buffer);
    caffe_add(bottom_size_, buffer, bottom_diff + (*bottom)[0]->offset(n),
        bottom_diff + (*bottom)[0]->offset(n));
  }

  // fill bottom_diff variance contribution term
  for (int n = 0; n < num_; ++n) {
    caffe_sub(bottom_size_, top_data + top[0]->offset(n), beta_data, buffer);
    caffe_mul(bottom_size_, buffer, variance_data, buffer);
    caffe_mul(bottom_size_, buffer, top_diff + top[0]->offset(n), buffer);
    caffe_add(bottom_size_, buffer, dl_dvar, dl_dvar);
  }
  caffe_powx(bottom_size_, variance_data, Dtype(-3.0), buffer);
  caffe_mul(bottom_size_, dl_dvar, buffer, dl_dvar);
  caffe_cpu_scale(bottom_size_, Dtype(-0.5), dl_dvar, dl_dvar);
  for (int n = 0; n < num_; ++n) {
    caffe_sub(bottom_size_, top_data + top[0]->offset(n), beta_data, buffer);
    caffe_div(bottom_size_, buffer, gamma_data, buffer);
    caffe_mul(bottom_size_, buffer, variance_data, buffer);
    caffe_cpu_scale(bottom_size_, Dtype(2) / Dtype(num_), buffer, buffer);
    caffe_mul(bottom_size_, buffer, dl_dvar, buffer);
    caffe_add(bottom_size_, buffer, bottom_diff + (*bottom)[0]->offset(n),
        bottom_diff + (*bottom)[0]->offset(n));
  }

  // fill bottom_diff mean contribution term
  for (int n = 0; n < num_; ++n) {
    caffe_mul(bottom_size_, top_diff + top[0]->offset(n), gamma_data, buffer);
    caffe_div(bottom_size_, buffer, variance_data, buffer);
    caffe_sub(bottom_size_, dl_dmean, buffer, dl_dmean);
  }
  caffe_cpu_scale(bottom_size_, Dtype(1) / Dtype(num_), dl_dmean, dl_dmean);
  for (int n = 0; n < num_; ++n) {
    caffe_add(bottom_size_, dl_dmean, bottom_diff + (*bottom)[0]->offset(n),
        bottom_diff + (*bottom)[0]->offset(n));
  }
}

#ifdef CPU_ONLY
STUB_GPU(BatchnormLayer);
#endif

INSTANTIATE_CLASS(BatchnormLayer);


}  // namespace caffe
