#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void XCovLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();

  // for now, we support only two inputs
  CHECK_EQ(bottom.size(), 2);

  for (int i = 0 ; i < bottom.size() ; i++) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    int num = bottom[i]->num();
    int dim = bottom[i]->count() / num;

    // calculate mean vector over batch
    caffe_gpu_gemv<Dtype>(CblasTrans, num, dim, 1. / num, bottom_data,
        batch_sum_multiplier_.gpu_data(), 0., mean_vec_[i]->mutable_gpu_data());

    // broadcast and negative mean vector
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
        batch_sum_multiplier_.gpu_data(),
        mean_vec_[i]->gpu_data(),
        0.,
        temp_vec_[i]->mutable_gpu_data());

    // subtract mean
    caffe_gpu_add(temp_vec_[i]->count(), bottom_data, temp_vec_[i]->gpu_data(),
        temp_vec_[i]->mutable_gpu_data());
  }

  int num = bottom[0]->num();
  int dim0 = bottom[0]->count() / num;
  int dim1 = bottom[1]->count() / num;

  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim0, dim1, num, 1./num,
      temp_vec_[0]->gpu_data(),
      temp_vec_[1]->gpu_data(),
      0.,
      xcov_.mutable_gpu_data());

  // square terms in xcov
  Dtype dot;
  caffe_gpu_dot<Dtype>(xcov_.count(), xcov_.gpu_data(), xcov_.gpu_data(), &dot);

  Dtype loss = dot / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void XCovLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype top_diff = top[0]->cpu_diff()[0];

  Dtype* bottom_diff_0 = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff_1 = bottom[1]->mutable_gpu_diff();

  int num = bottom[0]->num();
  int dim0 = bottom[0]->count() / num;
  int dim1 = bottom[1]->count() / num;

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, dim0, dim1, top_diff/num,
      temp_vec_[1]->gpu_data(),
      xcov_.gpu_data(),
      0.,
      bottom_diff_0);

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim1, dim0, top_diff/num,
      temp_vec_[0]->gpu_data(),
      xcov_.gpu_data(),
      0.,
      bottom_diff_1);
}


INSTANTIATE_LAYER_GPU_FUNCS(XCovLossLayer);


}  // namespace caffe
