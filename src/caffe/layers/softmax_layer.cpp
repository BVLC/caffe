#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  sum_multiplier_.Reshape(1, bottom[0]->channels(), 1, 1);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  for (int i = 0; i < sum_multiplier_.count(); ++i) {
    multiplier_data[i] = 1.;
  }
  scale_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count() / bottom[0]->num();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < num; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(spatial_dim, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < spatial_dim; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * spatial_dim + k]);
      }
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data + i * dim);
    // exponentiation
    caffe_exp<Dtype>(dim, top_data + i * dim, top_data + i * dim);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, 1.,
        top_data + i * dim, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(spatial_dim, top_data + (*top)[0]->offset(i, j), scale_data,
          top_data + (*top)[0]->offset(i, j));
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int num = top[0]->num();
  int channels = top[0]->channels();
  int dim = top[0]->count() / top[0]->num();
  int spatial_dim = top[0]->height() * top[0]->width();
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < num; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < spatial_dim; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, spatial_dim,
          top_data + i * dim + k, spatial_dim);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);


}  // namespace caffe
