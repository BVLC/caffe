#include <vector>
#include <stdio.h>
#include <cfloat>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DistanceLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "EM Layer takes two blobs as input.";

  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  // Figure out the dimensions for the difference
  difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  difference_squared_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  CHECK_EQ(top->size(), 1) << "EM Layer takes a single blob as output.";
  const int num_output = this->layer_param_.distance_param().num_output();
  bias_term_ = this->layer_param_.distance_param().bias_term();
  // Figure out the dimensions
  M_ = bottom[0]->num();
  K_ = bottom[0]->count() / bottom[0]->num();
  N_ = num_output;
  (*top)[0]->Reshape(bottom[0]->num(), num_output, 1, 1);

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.distance_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.distance_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  // Setting up the bias multiplier
  if (bias_term_) {
    bias_multiplier_.reset(new SyncedMemory(M_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < M_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
}

template <typename Dtype>
Dtype DistanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data_0 = bottom[0]->cpu_data();
  const Dtype* bottom_data_1 = bottom[1]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  int count = bottom[0]->count();

  caffe_sub(count, bottom_data_0, bottom_data_1,
      difference_.mutable_cpu_data());

  const Dtype* diff_data = difference_.cpu_data();
  const Dtype* diff_sq_data = difference_squared_.cpu_data();

  switch (this->layer_param_.distance_param().distance()) {
  case DistanceParameter_Distance_Squared:
    caffe_mul(count, diff_data, diff_data,
        difference_squared_.mutable_cpu_data());

    // project diff squared on w
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
        diff_sq_data, weight, (Dtype)0., top_data);

    break;
  case DistanceParameter_Distance_Abs:
  default:
    LOG(FATAL) << "Unknown Distance";
  }

  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);

    //LOG(ERROR) << "b: " << this->blobs_[1]->cpu_data()[0];
  }

  return Dtype(0);
}

template <typename Dtype>
void DistanceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* diff_sq_data = difference_squared_.cpu_data();
  const Dtype* diff_data = difference_.cpu_data();

  int count = (*bottom)[0]->count();

  switch (this->layer_param_.distance_param().distance()) {
  case DistanceParameter_Distance_Squared:
    // Gradient with respect to weight, squared diff
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, diff_sq_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
    break;
  case DistanceParameter_Distance_Abs:
  default:
    LOG(FATAL) << "Unknown Distance";
  }

  if (bias_term_) {
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), (Dtype)0.,
        this->blobs_[1]->mutable_cpu_diff());
  }

  if (propagate_down[0]) {
    switch (this->layer_param_.distance_param().distance()) {
    case DistanceParameter_Distance_Squared:

      // Gradient with respect to bottom data, squared diff
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)2.,
          top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
          (*bottom)[0]->mutable_cpu_diff());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)-2.,
          top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
          (*bottom)[1]->mutable_cpu_diff());

   
      caffe_mul(count, diff_data, (*bottom)[0]->cpu_diff(),
        (*bottom)[0]->mutable_cpu_diff());
      caffe_mul(count, diff_data, (*bottom)[1]->cpu_diff(),
        (*bottom)[1]->mutable_cpu_diff());

      break;
    case DistanceParameter_Distance_Abs:
    default:
      LOG(FATAL) << "Unknown Distance";
    }
  }
}

INSTANTIATE_CLASS(DistanceLayer);

}  // namespace caffe
