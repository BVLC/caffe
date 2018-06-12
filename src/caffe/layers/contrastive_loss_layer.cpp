#include <algorithm>
#include <vector>

#include "caffe/layers/contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ContrastiveLossLayer<Dtype, MItype, MOtype>::LayerSetUp(
  const vector<Blob<MItype>*>& bottom,
  const vector<Blob<MOtype>*>& top) {
  LossLayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_.Reshape(bottom[0]->shape(0), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->shape(0), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->shape(0), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int_tp i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);

  this->InitializeQuantizers(bottom, top);
  this->Reshape(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void ContrastiveLossLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  LossLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void ContrastiveLossLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  int_tp count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  const int_tp channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  Dtype loss(0.0);
  for (int_tp i = 0; i < bottom[0]->shape(0); ++i) {
    dist_sq_.mutable_cpu_data()[i] = caffe_dot(channels,
        diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
    if (static_cast<int_tp>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      if (legacy_version) {
        loss += fmax(margin - dist_sq_.cpu_data()[i], Dtype(0.0));
      } else {
        Dtype dist = std::max<Dtype>(margin - std::sqrt(dist_sq_.cpu_data()[i]),
          Dtype(0.0));
        loss += dist*dist;
      }
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->shape(0)) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype, typename MItype, typename MOtype>
void ContrastiveLossLayer<Dtype, MItype, MOtype>::Backward_cpu(const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom) {
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  for (int_tp i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->shape(0));
      int_tp num = bottom[i]->shape(0);
      int_tp channels = bottom[i]->channels();
      for (int_tp j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        if (static_cast<int_tp>(bottom[2]->cpu_data()[j])) {  // similar pairs
          caffe_axpby(
              channels,
              alpha,
              diff_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
        } else {  // dissimilar pairs
          Dtype mdist(0.0);
          Dtype beta(0.0);
          if (legacy_version) {
            mdist = margin - dist_sq_.cpu_data()[j];
            beta = -alpha;
          } else {
            Dtype dist = std::sqrt(dist_sq_.cpu_data()[j]);
            mdist = margin - dist;
            beta = -alpha * mdist / (dist + Dtype(1e-4));
          }
          if (mdist > Dtype(0.0)) {
            caffe_axpby(
                channels,
                beta,
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

INSTANTIATE_CLASS_3T_GUARDED(ContrastiveLossLayer,
                             (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(ContrastiveLossLayer,
                             (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(ContrastiveLossLayer,
                             (double), (double), (double));

REGISTER_LAYER_CLASS(ContrastiveLoss);
REGISTER_LAYER_CLASS_INST(ContrastiveLoss, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(ContrastiveLoss, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(ContrastiveLoss, (double), (double), (double));

}  // namespace caffe
