#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_pos.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[2]->gpu_data(),  // c
      diff_neg.mutable_gpu_data());  // a_i-c_i
  caffe_gpu_powx(
      count,
      diff_pos.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      diff_sq_pos.mutable_gpu_data());  // (a_i-b_i)^2
  caffe_gpu_powx(
      count,
      diff_neg.mutable_gpu_data(),  // a_i-c_i
      Dtype(2),
      diff_sq_neg.mutable_gpu_data());  // (a_i-c_i)^2
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  Dtype loss(0.0);
  // Loss component calculated from ab
  for (int i = 0; i < bottom[0]->num(); ++i) {
    /*dist_sq_pos.mutable_gpu_data()[i] = caffe_gpu_dot(channels,
        diff_pos.gpu_data() + (i*channels), diff_pos.gpu_data() + (i*channels));*/
    // ab is a similar pair
    dist_sq_.mutable_gpu_data()[i] = dist_sq_pos.gpu_data()[i];
    // Loss component calculated from ac
    /*dist_sq_neg.mutable_gpu_data()[i] = caffe_gpu_dot(channels,
        diff_neg.gpu_data() + (i*channels), diff_neg.gpu_data() + (i*channels));*/
    // ac is a dissimilar pair
    dist_sq_.mutable_gpu_data()[i] -= dist_sq_neg.gpu_data()[i];
    loss += std::max(margin + dist_sq_.gpu_data()[i], Dtype(0.0));
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_gpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
// there must be further check to ensure the gradient calc
    if (propagate_down[0]) {
      const Dtype sign = 1;
      const Dtype alpha = sign * top[0]->gpu_diff()[0] /
          static_cast<Dtype>(bottom[0]->num());
      int num = bottom[0]->num();
      int channels = bottom[0]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[0]->mutable_gpu_diff();
        if ((margin + dist_sq_.gpu_data()[j]) > Dtype(0.0)) {
        // similar pairs
          caffe_gpu_axpby(
              channels,
              alpha,
              diff_pos.gpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
        // dissimilar pairs
          caffe_gpu_axpby(
              channels,
              -alpha,
              diff_neg.gpu_data() + (j*channels),
              Dtype(1.0),
              bout + (j*channels));
        } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
        }
      }
    }
  for (int i = 1; i < 3; ++i) {
// there must be further check to ensure the gradient calc
    if (propagate_down[i]) {
      const Dtype sign = (i == 1) ? -1 : 1;
      const Dtype alpha = sign * top[0]->gpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_gpu_diff();
        if ((margin + dist_sq_.gpu_data()[j]) > Dtype(0.0)) {
          if (i == 1) {
        // similar pairs
          caffe_gpu_axpby(
              channels,
              alpha,
              diff_pos.gpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
          } else {
        // dissimilar pairs
          caffe_gpu_axpby(
              channels,
              alpha,
              diff_neg.gpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
          }
        } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);

}  // namespace caffe
