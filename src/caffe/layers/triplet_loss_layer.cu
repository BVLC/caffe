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
  caffe_gpu_sub(
      count,
      bottom[3]->gpu_data(),  // d
      bottom[4]->gpu_data(),  // e
      diff_par.mutable_gpu_data());  // d_i-e_i
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
  caffe_gpu_powx(
      count,
      diff_par.mutable_gpu_data(),  // d_i-e_i
      Dtype(2),
      diff_sq_par.mutable_gpu_data());  // (d_i-e_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_pos.gpu_data(),  // (a_i-b_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      dist_sq_pos.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_neg.gpu_data(),  // (a_i-c_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      dist_sq_neg.mutable_gpu_data());  // \Sum (a_i-c_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_par.gpu_data(),  // (a_i-c_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      dist_sq_par.mutable_gpu_data());  // \Sum (a_i-c_i)^2
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  Dtype losstype = this->layer_param_.triplet_loss_param().losstype();
  Dtype loss(0.0);
  
  if (losstype == 0) {
  for (int i = 0; i < bottom[0]->num(); ++i) {
    // Loss component calculated from ab
    // ab is a similar pair
    dist_sq_.mutable_gpu_data()[i] = dist_sq_pos.gpu_data()[i];
    // Loss component calculated from ac
    // ac is a dissimilar pair
    dist_sq_.mutable_gpu_data()[i] -= dist_sq_neg.gpu_data()[i];
    loss += std::max(margin + dist_sq_.gpu_data()[i], Dtype(0.0));
    // Pair wise loss accumulation
    // d e is a similar pair for pair wise
    // loss accumulated by the pair wise part
    loss += dist_sq_par.gpu_data()[i];
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_gpu_data()[0] = loss;
  } else {
  for (int i = 0; i < bottom[0]->num(); ++i) {
    // softTriplet loss accumulation
    // Loss component calculated from a and b
    // a b is a similar pair for triplet
    dist_sq_.mutable_gpu_data()[i] = dist_sq_pos.gpu_data()[i];
    dist_sq_.mutable_gpu_data()[i] += margin;
    // Loss component calculated from a and c
    // a c is a dissimilar pair for triplet
    dist_sq_.mutable_gpu_data()[i] = 1 - \
dist_sq_neg.gpu_data()[i] / dist_sq_.mutable_gpu_data()[i];
    // loss accumulated accumulated by the triplet part
    loss += std::max(dist_sq_.gpu_data()[i], Dtype(0.0));
    // Pair wise loss accumulation
    // d e is a similar pair for pair wise
    // loss accumulated by the pair wise part
    loss += dist_sq_par.gpu_data()[i];
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_gpu_data()[0] = loss;
  }
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  Dtype losstype = this->layer_param_.triplet_loss_param().losstype();
// there must be further check to ensure the gradient calc
  if (losstype == 0) {
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
  } else {
    // BP for data1(feat1)
    if (propagate_down[0]) {
      const Dtype alpha = top[0]->gpu_diff()[0] /
          static_cast<Dtype>(bottom[0]->num());
      int num = bottom[0]->num();
      int channels = bottom[0]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[0]->mutable_gpu_diff();
        if ((dist_sq_.gpu_data()[j]) > Dtype(0.0)) {
          caffe_gpu_axpby(
              channels,
              alpha*dist_sq_neg.mutable_gpu_data()[j]\
/((dist_sq_pos.mutable_gpu_data()[j]+margin)\
*(dist_sq_pos.mutable_gpu_data()[j]+margin)),
              diff_pos.gpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
          caffe_gpu_axpby(
              channels,
              -alpha*(dist_sq_pos.mutable_gpu_data()[j] + margin)\
/((dist_sq_pos.mutable_gpu_data()[j] + margin)\
*(dist_sq_pos.mutable_gpu_data()[j] + margin)),
              diff_neg.gpu_data() + (j*channels),
              Dtype(1.0),
              bout + (j*channels));
        } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
        }
      }
    }
    // BP for positive data(feat2)
    if (propagate_down[1]) {
      const Dtype alpha = top[0]->gpu_diff()[0] /
          static_cast<Dtype>(bottom[1]->num());
      int num = bottom[1]->num();
      int channels = bottom[1]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[1]->mutable_gpu_diff();
        if ((dist_sq_.gpu_data()[j]) > Dtype(0.0)) {
          caffe_gpu_axpby(
              channels,
              -alpha*dist_sq_neg.mutable_gpu_data()[j]\
/((dist_sq_pos.mutable_gpu_data()[j] + margin)\
*(dist_sq_pos.mutable_gpu_data()[j] + margin)),
              diff_pos.gpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
        } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
        }
      }
    }
    // BP for negative data(feat3)
    if (propagate_down[2]) {
      const Dtype alpha = top[0]->gpu_diff()[0] /
          static_cast<Dtype>(bottom[2]->num());
      int num = bottom[2]->num();
      int channels = bottom[2]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[2]->mutable_gpu_diff();
        if ((dist_sq_.gpu_data()[j]) > Dtype(0.0)) {
          caffe_gpu_axpby(
              channels,
              alpha/(dist_sq_pos.mutable_gpu_data()[j] + margin),
              diff_neg.gpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
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
      const Dtype alpha = sign * top[0]->gpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_gpu_diff();  // similar pairs
          caffe_gpu_axpby(
              channels,
              alpha,
              diff_par.gpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);

}  // namespace caffe
