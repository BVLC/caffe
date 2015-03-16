#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void XCovLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  mean_vec_.clear();
  mean_vec_.push_back(&mean_0_);
  mean_vec_.push_back(&mean_1_);

  temp_vec_.clear();
  temp_vec_.push_back(&temp_0_);
  temp_vec_.push_back(&temp_1_);
}


template <typename Dtype>
void XCovLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  int num = bottom[0]->num();
  top[0]->Reshape(1, 1, 1, 1);

  int dim0 = bottom[0]->count() / num;
  int dim1 = bottom[1]->count() / num;
  xcov_.Reshape(dim0*dim1, 1, 1, 1);

  for (int i = 0 ; i < bottom.size() ; i++ ) {
    mean_vec_[i]->Reshape(1, bottom[i]->channels(),
        bottom[i]->height(), bottom[i]->width());
    temp_vec_[i]->Reshape(bottom[i]->num(), bottom[i]->channels(),
        bottom[i]->height(), bottom[i]->width());
  }
  batch_sum_multiplier_.Reshape(bottom[0]->num(), 1, 1, 1);
  Dtype* batch_multiplier_data = batch_sum_multiplier_.mutable_cpu_data();
  caffe_set(batch_sum_multiplier_.count(), Dtype(1), batch_multiplier_data);
}

template <typename Dtype>
void XCovLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // for now, we support only two inputs
  CHECK_EQ(bottom.size(), 2);

  for (int i = 0 ; i < bottom.size() ; i++) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    int num = bottom[i]->num();
    int dim = bottom[i]->count() / num;

    // calculate mean vector over batch
    caffe_cpu_gemv<Dtype>(CblasTrans, num, dim, 1. / num, bottom_data,
        batch_sum_multiplier_.cpu_data(), 0., mean_vec_[i]->mutable_cpu_data());

    // broadcast and negative mean vector
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
        batch_sum_multiplier_.cpu_data(),
        mean_vec_[i]->cpu_data(),
        0.,
        temp_vec_[i]->mutable_cpu_data());

    // subtract mean
    caffe_add(temp_vec_[i]->count(), bottom_data, temp_vec_[i]->cpu_data(),
        temp_vec_[i]->mutable_cpu_data());
  }

  int num = bottom[0]->num();
  int dim0 = bottom[0]->count() / num;
  int dim1 = bottom[1]->count() / num;

  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim0, dim1, num, 1./num,
      temp_vec_[0]->cpu_data(),
      temp_vec_[1]->cpu_data(),
      0.,
      xcov_.mutable_cpu_data());

  // square terms in xcov
  Dtype dot = caffe_cpu_dot<Dtype>(xcov_.count(), xcov_.cpu_data(), xcov_.cpu_data());

  Dtype loss = dot / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void XCovLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype top_diff = top[0]->cpu_diff()[0];

  Dtype* bottom_diff_0 = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_diff_1 = bottom[1]->mutable_cpu_diff();

  int num = bottom[0]->num();
  int dim0 = bottom[0]->count() / num;
  int dim1 = bottom[1]->count() / num;

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, dim0, dim1, top_diff/num,
      temp_vec_[1]->cpu_data(),
      xcov_.cpu_data(),
      0.,
      bottom_diff_0);

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim1, dim0, top_diff/num,
      temp_vec_[0]->cpu_data(),
      xcov_.cpu_data(),
      0.,
      bottom_diff_1);
}


#ifdef CPU_ONLY
STUB_GPU(XCovLossLayer);
#endif

INSTANTIATE_CLASS(XCovLossLayer);
REGISTER_LAYER_CLASS(XCovLoss);

}  // namespace caffe
