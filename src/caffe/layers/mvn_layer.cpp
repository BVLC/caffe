#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/device.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void MVNLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  mean_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  variance_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  sum_multiplier_.Reshape(1, 1,
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_data();
  this->device_->set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
Dtype MVNLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->const_data();
  Dtype* top_data = (*top)[0]->mutable_data();
  int num;
  if (this->layer_param_.mvn_param().across_channels()) {
    num = bottom[0]->num();
  } else {
    num = bottom[0]->num() * bottom[0]->channels();
  }

  int dim = bottom[0]->count() / num;
  Dtype eps = 1e-10;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    // put the squares of bottom into temp_
    this->device_->powx(bottom[0]->count(), bottom_data, Dtype(2),
        temp_.mutable_data());

    // computes variance using var(X) = E(X^2) - (EX)^2
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, temp_.const_data(),
        sum_multiplier_.const_data(), 0.,
        variance_.mutable_data());  // E(X^2)
    this->device_->powx(mean_.count(), mean_.const_data(), Dtype(2),
        temp_.mutable_data());  // (EX)^2
    this->device_->sub(mean_.count(), variance_.const_data(),
        temp_.const_data(), variance_.mutable_data());  // variance

    // do mean and variance normalization
    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
        mean_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());

    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
  }

  return Dtype(0);
}

template <typename Dtype>
void MVNLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->const_diff();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* bottom_data = (*bottom)[0]->const_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels()) {
    num = (*bottom)[0]->num();
  } else {
    num = (*bottom)[0]->num() * (*bottom)[0]->channels();
  }

  int dim = (*bottom)[0]->count() / num;
  Dtype eps = 1e-10;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    this->device_->mul(temp_.count(), top_data, top_diff, bottom_diff);
    this->device_->gemv(CblasNoTrans, num, dim, 1., bottom_diff,
          sum_multiplier_.const_data(), 0., mean_.mutable_data());
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          mean_.const_data(), sum_multiplier_.const_data(), 0.,
          bottom_diff);
    this->device_->mul(temp_.count(), top_data, bottom_diff, bottom_diff);

    this->device_->gemv(CblasNoTrans, num, dim, 1., top_diff,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_.const_data(), sum_multiplier_.const_data(), 1.,
            bottom_diff);

    this->device_->axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim),
        bottom_diff);

    // put the squares of bottom into temp_
    this->device_->powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_data());

    // computes variance using var(X) = E(X^2) - (EX)^2
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, temp_.const_data(),
        sum_multiplier_.const_data(), 0.,
        variance_.mutable_data());  // E(X^2)
    this->device_->powx(mean_.count(), mean_.const_data(), Dtype(2),
        temp_.mutable_data());  // (EX)^2
    this->device_->sub(mean_.count(), variance_.const_data(),
        temp_.const_data(), variance_.mutable_data());  // variance

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

    this->device_->add_scalar(variance_.count(), eps, variance_.mutable_data());

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());

    this->device_->div(temp_.count(), bottom_diff, temp_.const_data(),
        bottom_diff);
  } else {
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
  }
}


INSTANTIATE_CLASS(MVNLayer);


}  // namespace caffe
