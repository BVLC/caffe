#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe {

#define sign(x) (Dtype(0) < (x)) - ((x) < Dtype(0))

template <typename Dtype>
void NormalizeLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  normalize_type_ =
    this->layer_param_.normalize_param().normalize_type();
  rescale_ =
    this->layer_param_.normalize_param().rescale();
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Reshape_const(bottom,top);
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape_const(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  if (top.size() == 2) {
    top[1]->Reshape(bottom[0]->num(), 1,
                    bottom[0]->height(), bottom[0]->width());
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_const_cpu(bottom,top);
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_const_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) const {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Blob<Dtype> squared,norm;
  squared.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  Dtype* square_data = squared.mutable_cpu_data();

  if (top.size() != 2) {
    norm.Reshape(bottom[0]->num(), 1,
                   bottom[0]->height(), bottom[0]->width());
  }

  Dtype* norm_data = (top.size() == 2) ? top[1]->mutable_cpu_data() : norm.mutable_cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  if (normalize_type_ == "L2") {
    caffe_sqr<Dtype>(num*channels*spatial_dim, bottom_data, square_data);
    for (int n = 0; n < num; n++) {
      for (int s = 0; s < spatial_dim; s++) {
        norm_data[n*spatial_dim + s] = Dtype(0);
        for (int c = 0; c < channels; c++) {
          norm_data[n*spatial_dim + s] += square_data[(n * channels + c) * spatial_dim + s];
        }
        norm_data[n*spatial_dim + s] += 1e-6;
        norm_data[n*spatial_dim + s] = Dtype(1) / sqrt(norm_data[n*spatial_dim + s]);
        for (int c = 0; c < channels; c++) {
          top_data[(n * channels + c) * spatial_dim + s] = bottom_data[(n * channels + c) * spatial_dim + s] * norm_data[n*spatial_dim + s];
        }
      }
    }
  }
  else if (normalize_type_ == "L1") {
    caffe_abs<Dtype>(num*channels*spatial_dim, bottom_data, square_data);
    for (int n = 0; n < num; n++) {
      for (int s = 0; s < spatial_dim; s++) {
        norm_data[n*spatial_dim +s] = Dtype(0);
        for (int c = 0; c < channels; c++) {
          norm_data[n*spatial_dim + s] += square_data[(n * channels + c) * spatial_dim + s];
        }
        norm_data[n*spatial_dim + s] += 1e-6;
        norm_data[n*spatial_dim + s] = Dtype(1) / norm_data[n*spatial_dim + s];
        for (int c = 0; c < channels; c++) {
          top_data[(n * channels + c) * spatial_dim + s] = bottom_data[(n * channels + c) * spatial_dim + s] * norm_data[n*spatial_dim + s];
        }
      }
    }
  }
  else {
    NOT_IMPLEMENTED;
  }
}


#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe
