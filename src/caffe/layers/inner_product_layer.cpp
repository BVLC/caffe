#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void InnerProductLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const int_tp num_output =
      this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int_tp axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, n inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO)<< "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int_tp> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape, this->device_));
    // fill the weights (for float types only)
    if (is_float_type<Dtype>()) {
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.inner_product_param().weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());
    }
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int_tp> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape, this->device_));
      if (is_float_type<Dtype>()) {
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                this->layer_param_.inner_product_param().bias_filler()));
        bias_filler->Fill(this->blobs_[1].get());
      }
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void InnerProductLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  // Figure out the dimensions
  const int_tp axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int_tp new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int_tp> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int_tp> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    bias_multiplier_qv_.scale = 1.0;
    bias_multiplier_qv_.zero = 0.0;
    bias_multiplier_qv_.one = 1.0;
    bias_multiplier_qv_.max = 1.0;
    bias_multiplier_qv_.min = 0.0;
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void InnerProductLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  caffe_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, Dtype(1), bottom_data, weight, Dtype(0), top_data,
      nullptr, &(this->bottom_quants_[0]->out_quantizer_values()),
      &(this->blobs_quants_[0]->out_quantizer_values()),
      nullptr, &(this->top_quants_[0]->in_quantizer_values()));

  if (bias_term_) {
    caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, Dtype(1),
                      bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(),
                      Dtype(1), top_data,
                      nullptr, &bias_multiplier_qv_,
                      &(this->blobs_quants_[1]->out_quantizer_values()),
                      nullptr, &(this->top_quants_[0]->in_quantizer_values()));
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void InnerProductLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_gemv<Dtype>(CblasTrans, M_, N_, (Dtype) 1., top_diff,
                          bias_multiplier_.cpu_data(), (Dtype) 1.,
                          this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(InnerProductLayer,
                             (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(InnerProductLayer,
                             (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(InnerProductLayer,
                             (double), (double), (double));
INSTANTIATE_CLASS_3T_GUARDED(InnerProductLayer,
                             (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASS_3T_GUARDED(InnerProductLayer,
                             (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASS_3T_GUARDED(InnerProductLayer,
                             (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASS_3T_GUARDED(InnerProductLayer,
                             (uint64_t), (uint64_t), (uint64_t));

REGISTER_LAYER_CLASS(InnerProduct);
REGISTER_LAYER_CLASS_INST(InnerProduct, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(InnerProduct, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(InnerProduct, (double), (double), (double));
REGISTER_LAYER_CLASS_INST(InnerProduct, (uint8_t), (uint8_t), (uint8_t));
REGISTER_LAYER_CLASS_INST(InnerProduct, (uint16_t), (uint16_t), (uint16_t));
REGISTER_LAYER_CLASS_INST(InnerProduct, (uint32_t), (uint32_t), (uint32_t));
REGISTER_LAYER_CLASS_INST(InnerProduct, (uint64_t), (uint64_t), (uint64_t));


}  // namespace caffe
