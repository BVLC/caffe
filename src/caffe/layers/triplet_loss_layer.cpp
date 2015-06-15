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
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  CHECK_EQ(bottom[3]->channels(), 1);
  CHECK_EQ(bottom[3]->height(), 1);
  CHECK_EQ(bottom[3]->width(), 1);
  diff_pos.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_neg.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_pos.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_neg.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  dist_sq_pos.Reshape(bottom[0]->num(), 1, 1, 1);
  dist_sq_neg.Reshape(bottom[0]->num(), 1, 1, 1);
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
      diff_pos.mutable_cpu_data());  // a_i-b_i
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[2]->cpu_data(),  // c
      diff_neg.mutable_cpu_data());  // a_i-c_i
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  Dtype loss(0.0);

  // Loss component calculated from ab
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_pos.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_pos.cpu_data() + (i*channels), diff_pos.cpu_data() + (i*channels));
    // ab is a similar pair
    dist_sq_.mutable_cpu_data()[i] += dist_sq_pos.cpu_data()[i];
    // Loss component calculated from ac
    dist_sq_neg.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_neg.cpu_data() + (i*channels), diff_neg.cpu_data() + (i*channels));
    // ac is a dissimilar pair
    dist_sq_.mutable_cpu_data()[i] -= dist_sq_neg.cpu_data()[i];
    loss += std::max(margin + dist_sq_.cpu_data()[i], Dtype(0.0));
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  for (int i = 1; i < 3; ++i) {
// there must be further check to ensure the gradient calc
    if (propagate_down[i]) {
      const Dtype sign = (i == 2) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {	
        Dtype* bout = bottom[i]->mutable_cpu_diff();
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
              Dtype(0.0),
              bout + (j*channels));
	  } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
          }
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
