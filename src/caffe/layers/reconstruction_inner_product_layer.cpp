// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReconstructionInnerProductLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Reconstruction IP Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Reconstruction IP Layer takes a single blob as output.";
  const int num_output = this->layer_param_.reconstruction_inner_product_param().num_output();
  // Reconsturction inner product does not support bias
  //bias_term_ = this->layer_param_.reconstruction_inner_product_param().bias_term(); 
  // Figure out the dimensions
  M_ = bottom[0]->num();
  K_ = bottom[0]->count() / bottom[0]->num();
  N_ = num_output;
  (*top)[0]->Reshape(bottom[0]->num(), num_output, 1, 1);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.reconstruction_inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->difference_.Reshape(M_, bottom[0]->channels(), 
         bottom[0]->width(), bottom[0]->height());
  this->x_Tx_.Reshape(K_*K_, 1, 1, 1);
  this->w_Tw_.Reshape(K_*K_, 1, 1, 1);
  this->x_Txw_Tw_.Reshape(K_*K_, 1, 1, 1);
}

template <typename Dtype>
Dtype ReconstructionInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);

  // compute the reconstruction error and return that as loss value.
  caffe_copy<Dtype>(bottom[0]->count(), bottom_data, difference_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
      top_data, weight, (Dtype)-1., difference_.mutable_cpu_data());
  return caffe_cpu_dot<Dtype>(difference_.count(), difference_.cpu_data(),
           difference_.cpu_data());
}

template <typename Dtype>
void ReconstructionInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  // Gradient with respect to weight
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
      top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
  
  // now add the reconstruction cost's gradient

  const Dtype* weight = this->blobs_[0]->cpu_data();
  // compute W^T W
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, K_, N_, (Dtype)1., 
      weight, weight, (Dtype)0., w_Tw_.mutable_cpu_data()); 
  
  // compute X^T X
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, K_, M_, (Dtype)1.,
      bottom_data, bottom_data, (Dtype)0., x_Tx_.mutable_cpu_data()); 
  
  // compute X^T X W^T W
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, K_, K_, (Dtype)1.,
      x_Tx_.cpu_data(), w_Tw_.cpu_data(), (Dtype)0., 
      x_Txw_Tw_.mutable_cpu_data()); 

  // now add 2 W X^T X W^T W to the weight diff
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, K_, K_, (Dtype)2.,
      weight, x_Txw_Tw_.cpu_data(), (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  
  // now add 2 W W^T W X^T X to the weight diff.
  // Note that W^T W X^T X is the transpose of X^T X W^T W
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, K_, K_, (Dtype)2.,
      weight, x_Txw_Tw_.cpu_data(), (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  
  // now add -4 W X^T X to weight diff
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, K_, K_, (Dtype)-4.,
      weight, x_Tx_.cpu_data(), (Dtype)1., this->blobs_[0]->mutable_cpu_diff());

  if (propagate_down) {
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        (*bottom)[0]->mutable_cpu_diff());

    // now add the gradient of the reconstruction term w.r.t. bottom
    // add 2 X to the mutable_cpu_diff
    caffe_cpu_axpby<Dtype>((*bottom)[0]->count(), 
        (Dtype)2., (*bottom)[0]->cpu_data(),
        (Dtype)1., (*bottom)[0]->mutable_cpu_diff());
    // compute -4 W^T W + 2 W^T W W^T W
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, K_, K_, 
        (Dtype)2., w_Tw_.cpu_data(), w_Tw_.cpu_data(), (Dtype)-4.,
        w_Tw_.mutable_cpu_data()); //note w_Tw_ is no longer W^T W :(
    // multiply the above thing by X and add it to mutable_cpu_diff
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, K_, 
        (Dtype)1., (*bottom)[0]->cpu_data(), w_Tw_.cpu_data(),
        (Dtype)1., (*bottom)[0]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(ReconstructionInnerProductLayer);

}  // namespace caffe
