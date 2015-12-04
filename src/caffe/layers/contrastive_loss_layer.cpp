#include <algorithm>
#include <vector>

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
#include "caffe/loss_layers.hpp"
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
#include "caffe/loss_layers.hpp"
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
#include "caffe/loss_layers.hpp"
=======
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/loss_layers.hpp"
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/loss_layers.hpp"
=======
>>>>>>> pod/caffe-merge
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
>>>>>>> origin/BVLC/parallel
=======
#include "caffe/loss_layers.hpp"
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/loss_layers.hpp"
>>>>>>> device-abstraction
=======
#include "caffe/layers/contrastive_loss_layer.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/layers/contrastive_loss_layer.hpp"
>>>>>>> BVLC/master
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod/device/blob.hpp
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod/caffe-merge
>>>>>>> origin/BVLC/parallel
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
      if (legacy_version) {
        loss += std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0));
      } else {
        Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq_.cpu_data()[i]),
          Dtype(0.0));
        loss += dist*dist;
      }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
      loss += std::max(margin-dist_sq_.cpu_data()[i], Dtype(0.0));
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod/device/blob.hpp
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod/caffe-merge
>>>>>>> origin/BVLC/parallel
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
        } else {  // dissimilar pairs
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
          Dtype mdist(0.0);
          Dtype beta(0.0);
          if (legacy_version) {
            mdist = margin - dist_sq_.cpu_data()[j];
            beta = -alpha;
          } else {
            Dtype dist = sqrt(dist_sq_.cpu_data()[j]);
            mdist = margin - dist;
            beta = -alpha * mdist / (dist + Dtype(1e-4));
          }
          if (mdist > Dtype(0.0)) {
            caffe_cpu_axpby(
                channels,
                beta,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
          if ((margin-dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
            caffe_cpu_axpby(
                channels,
                -alpha,
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
                diff_.cpu_data() + (j*channels),
                Dtype(0.0),
                bout + (j*channels));
          } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ContrastiveLossLayer);
#endif

INSTANTIATE_CLASS(ContrastiveLossLayer);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
REGISTER_LAYER_CLASS(ContrastiveLoss);

=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
REGISTER_LAYER_CLASS(ContrastiveLoss);

=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
REGISTER_LAYER_CLASS(ContrastiveLoss);

=======
>>>>>>> pod/device/blob.hpp
=======
REGISTER_LAYER_CLASS(ContrastiveLoss);

=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
REGISTER_LAYER_CLASS(ContrastiveLoss);

=======
>>>>>>> pod/caffe-merge
REGISTER_LAYER_CLASS(CONTRASTIVE_LOSS, ContrastiveLossLayer);
>>>>>>> origin/BVLC/parallel
=======
REGISTER_LAYER_CLASS(ContrastiveLoss);

>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
REGISTER_LAYER_CLASS(ContrastiveLoss);

>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
}  // namespace caffe
