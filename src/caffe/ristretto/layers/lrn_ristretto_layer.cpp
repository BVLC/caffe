#include <vector>

#include "caffe/layers/lrn_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
LRNRistrettoLayer<Dtype>::LRNRistrettoLayer(const LayerParameter& param)
      : LRNLayer<Dtype>(param), BaseRistrettoLayer<Dtype>() {
  this->precision_ = this->layer_param_.quantization_param().precision();
  this->rounding_ = this->layer_param_.quantization_param().rounding_scheme();
  switch (this->precision_) {
  case QuantizationParameter_Precision_FIXED_POINT:
    LOG(ERROR) << "LRN layer only supports mini floating point";
    break;
  case QuantizationParameter_Precision_MINI_FLOATING_POINT:
    this->fp_mant_ = this->layer_param_.quantization_param().mant_bits();
    this->fp_exp_ = this->layer_param_.quantization_param().exp_bits();
    break;
  case QuantizationParameter_Precision_POWER_2_WEIGHTS:
    LOG(ERROR) << "LRN layer only supports mini floating point";
    break;
  default:
    LOG(FATAL) << "Unknown precision mode: " << this->precision_;
    break;
  }
}

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //TODO
  LOG(ERROR) << "LRNRistrettoLayer not implemented on CPU yet.";
}

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //TODO
  LOG(ERROR) << "LRNRistrettoLayer not implemented on CPU yet.";
}

#ifdef CPU_ONLY
STUB_GPU(LRNRistrettoLayer);
STUB_GPU_FORWARD(LRNRistrettoLayer, CrossChannelForward);
STUB_GPU_BACKWARD(LRNRistrettoLayer, CrossChannelBackward);
#endif

INSTANTIATE_CLASS(LRNRistrettoLayer);
REGISTER_LAYER_CLASS(LRNRistretto);

}  // namespace caffe

