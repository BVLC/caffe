#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[0]->channels(), bottom[3]->channels());
  CHECK_EQ(bottom[0]->channels(), bottom[4]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  CHECK_EQ(bottom[3]->height(), 1);
  CHECK_EQ(bottom[3]->width(), 1);
  CHECK_EQ(bottom[4]->height(), 1);
  CHECK_EQ(bottom[4]->width(), 1);
  CHECK_EQ(bottom[5]->channels(), 1);
  CHECK_EQ(bottom[5]->height(), 1);
  CHECK_EQ(bottom[5]->width(), 1);
  diff_pos.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_neg.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_par.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_pos.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_neg.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_par.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  dist_sq_pos.Reshape(bottom[0]->num(), 1, 1, 1);
  dist_sq_neg.Reshape(bottom[0]->num(), 1, 1, 1);
  dist_sq_par.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_pos.mutable_cpu_data());  // a_i-b_i: positive
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[2]->cpu_data(),  // c
      diff_neg.mutable_cpu_data());  // a_i-c_i: negative
  caffe_sub(
      count,
      bottom[3]->cpu_data(),  // d
      bottom[4]->cpu_data(),  // e
      diff_par.mutable_cpu_data());  // d_i-e_i: pair wise
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  Dtype loss(0.0);

  for (int i = 0; i < bottom[0]->num(); ++i) {
    // Triplet loss accumulation
    // Loss component calculated from a and b
    dist_sq_pos.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_pos.cpu_data() + (i*channels), diff_pos.cpu_data() + (i*channels));
    // a b is a similar pair for triplet
    dist_sq_.mutable_cpu_data()[i] = dist_sq_pos.cpu_data()[i];
    // Loss component calculated from a and c
    dist_sq_neg.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_neg.cpu_data() + (i*channels), diff_neg.cpu_data() + (i*channels));
    // a c is a dissimilar pair for triplet
    dist_sq_.mutable_cpu_data()[i] -= dist_sq_neg.cpu_data()[i];
    // loss accumulated accumulated by the triplet part
    loss += std::max(margin + dist_sq_.cpu_data()[i], Dtype(0.0));
    // Pair wise loss accumulation
    // Loss component calculated from d and e
    dist_sq_par.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_par.cpu_data() + (i*channels), diff_par.cpu_data() + (i*channels));
    // d e is a similar pair for pair wise
    // loss accumulated by the pair wise part
    loss += dist_sq_par.cpu_data()[i];
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
    if (propagate_down[0]) {
      const Dtype sign = 1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[0]->num());
      int num = bottom[0]->num();
      int channels = bottom[0]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[0]->mutable_cpu_diff();
        if ((margin + dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
        // similar pairs
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_pos.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
        // dissimilar pairs
          caffe_cpu_axpby(
              channels,
              -alpha,
              diff_neg.cpu_data() + (j*channels),
              Dtype(1.0),
              bout + (j*channels));
        } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
        }
      }
    }
  for (int i = 1; i < 3; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 1) ? -1 : 1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        if ((margin + dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
          if (i == 1) {
        // similar pairs
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_pos.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
          } else {
        // dissimilar pairs
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_neg.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
          }
        } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
        }
      }
    }
  }
  // pair wise back
  for (int i = 3; i < 5; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 3) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();  // similar pairs
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_par.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe
