#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu_fast_case(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* mult = sum_multiplier_.cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  // assert(dim == channels);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < outer_num_; ++i) {
    const Dtype* bottom_data = bottom[0]->cpu_data() + i*dim;
    Dtype *top_data = top[0]->mutable_cpu_data() + channels*i;

    caffe_copy(dim, bottom_data, top_data);

    Dtype scale_data = bottom_data[0];
    for (int j = 1; j < channels; j++) {
        scale_data = std::max(scale_data, bottom_data[j]);
    }

    // subtraction
    for (int j = 0; j < channels; ++j)
        top_data[j] -= scale_data*mult[j];

    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);

    // sum after exp
    scale_data = 0;
    for (int j = 0; j < channels; ++j)
        scale_data += top_data[j]*mult[j];

    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data + j, &scale_data, top_data + j);
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;

  if (inner_num_ == 1 && channels == dim) {
      Forward_cpu_fast_case(bottom, top);
      return;
  }

  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int k = 0; k < inner_num_; k++) {
      Dtype max_val = bottom_data[i * dim + k];
      for (int j = 1; j < channels; j++) {
        Dtype value = bottom_data[i * dim + k + j * inner_num_];
        if (max_val < value) max_val = value;
      }
      scale_data[k] = max_val;
    }

    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data + j*inner_num_, scale_data,
              top_data + j*inner_num_);
    }
    top_data += channels*inner_num_;
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
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
