#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
BaseRistrettoLayer<Dtype>::BaseRistrettoLayer(){
  // Initialize random number generator
  srand(time(NULL));
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeWeights_cpu(Dtype* weight,
      const int cnt_weight, Dtype* bias, const int cnt_bias, const int rounding) {
  switch (this->precision_) {
  case QuantizationParameter_Precision_MINI_FLOATING_POINT:
    this->Trim2FloatingPoint_cpu(weight, cnt_weight, this->fp_mant_,
        this->fp_exp_, rounding);
    this->Trim2FloatingPoint_cpu(bias, cnt_bias, this->fp_mant_, this->fp_exp_,
        rounding);
    break;
  case QuantizationParameter_Precision_FIXED_POINT:
    this->Trim2FixedPoint_cpu(weight, cnt_weight, this->bw_params_, rounding,
        this->fl_params_);
    this->Trim2FixedPoint_cpu(bias, cnt_bias, this->bw_params_, rounding,
        this->fl_params_);
    break;
  case QuantizationParameter_Precision_POWER_2_WEIGHTS:
    this->Trim2PowerOf2_cpu(weight, cnt_weight, this->pow_2_min_exp_,
        this->pow_2_max_exp_, rounding);
    // Don't trim bias
    break;
  default:
    LOG(FATAL) << "Unknown trimming mode: " << this->precision_;
    break;
  }
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeLayerOutputs_cpu(
      Dtype* top_data, const int top_count){
  switch (this->precision_) {
    case QuantizationParameter_Precision_POWER_2_WEIGHTS:
      break;
    case QuantizationParameter_Precision_FIXED_POINT:
      this->Trim2FixedPoint_cpu(top_data, top_count, this->bw_layer_out_,
          this->rounding_, this->fl_layer_out_);
      break;
    case QuantizationParameter_Precision_MINI_FLOATING_POINT:
      this->Trim2FloatingPoint_cpu(top_data, top_count, this->fp_mant_,
          this->fp_exp_, this->rounding_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << this->precision_;
      break;
  }
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FixedPoint_cpu(Dtype* data, const int cnt,
      const int bit_width, const int rounding, int fl) {
  for (int index = 0; index < cnt; ++index) {
    // Saturate data
    Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
    Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
    data[index] = std::max(std::min(data[index], max_data), min_data);
    // Round data
    data[index] /= pow(2, -fl);
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:{
      data[index] = round(data[index]);
      break;}
    case QuantizationParameter_Rounding_STOCHASTIC:{
      data[index] = floor(data[index] + RandUniform_cpu());
      break;}
    default:{
      break;}
    }
    data[index] *= pow(2, -fl);
	}
}

typedef union {
  float d;
  struct {
    unsigned int mantisa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float_cast;

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FloatingPoint_cpu(Dtype* data,
      const int cnt, const int bw_mant, const int bw_exp, const int rounding) {
  for (int index = 0; index < cnt; ++index) {
    int bias_out = pow(2, bw_exp - 1) - 1;
    float_cast d2;
    // This casts the input to single precision
    d2.d = (float)data[index];
    int exponent=d2.parts.exponent - 127 + bias_out;
    double mantisa = d2.parts.mantisa;
    // Special case: input is zero or denormalized number
    if (d2.parts.exponent == 0) {
      data[index] = 0;
      return;
    }
    // Special case: denormalized number as output
    if (exponent < 0) {
      data[index] = 0;
      return;
    }
    // Saturation: input float is larger than maximum output float
    int max_exp = pow(2, bw_exp) - 1;
    int max_mant = pow(2, bw_mant) - 1;
    if (exponent > max_exp) {
      exponent = max_exp;
      mantisa = max_mant;
    } else{
      // Convert mantissa from long format to short one. Cut off LSBs.
      double tmp = mantisa / pow(2, 23 - bw_mant);
      switch (rounding) {
      case QuantizationParameter_Rounding_NEAREST:{
        mantisa = round(tmp);
        break;}
      case QuantizationParameter_Rounding_STOCHASTIC:{
        mantisa = floor(tmp + RandUniform_cpu());
        break;}
      default:{
        break;}
      }
    }
    // Assemble result
    data[index] = pow(-1, d2.parts.sign) * ((mantisa + pow(2, bw_mant)) /
        pow(2, bw_mant)) * pow(2, exponent - bias_out);
	}
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2PowerOf2_cpu(Dtype* data, const int cnt,
      const int min_exp, const int max_exp, const int rounding) {
	for (int index = 0; index < cnt; ++index) {
    float exponent = log2f((float)fabs(data[index]));
    int sign = data[index] >= 0 ? 1 : -1;
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      exponent = round(exponent);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:{
      exponent = floorf(exponent + RandUniform_cpu());
      break;}
    default:
      break;
    }
    exponent = std::max(std::min(exponent, (float)max_exp), (float)min_exp);
    data[index] = sign * pow(2, exponent);
	}
}

template <typename Dtype>
double BaseRistrettoLayer<Dtype>::RandUniform_cpu(){
  return rand() / (RAND_MAX+1.0);
}

template BaseRistrettoLayer<double>::BaseRistrettoLayer();
template BaseRistrettoLayer<float>::BaseRistrettoLayer();
template void BaseRistrettoLayer<double>::QuantizeWeights_cpu(double* weight,
    const int cnt_weight, double* bias, const int cnt_bias, const int rounding);
template void BaseRistrettoLayer<float>::QuantizeWeights_cpu(float* weight,
    const int cnt_weight, float* bias, const int cnt_bias, const int rounding);
template void BaseRistrettoLayer<double>::QuantizeLayerOutputs_cpu(
    double* top_data, const int top_count);
template void BaseRistrettoLayer<float>::QuantizeLayerOutputs_cpu(
    float* top_data, const int top_count);
template void BaseRistrettoLayer<double>::Trim2FixedPoint_cpu(double* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseRistrettoLayer<float>::Trim2FixedPoint_cpu(float* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseRistrettoLayer<double>::Trim2FloatingPoint_cpu(double* data,
    const int cnt, const int bw_mant, const int bw_exp, const int rounding);
template void BaseRistrettoLayer<float>::Trim2FloatingPoint_cpu(float* data,
    const int cnt, const int bw_mant, const int bw_exp, const int rounding);
template void BaseRistrettoLayer<double>::Trim2PowerOf2_cpu(double* data,
    const int cnt, const int min_exp, const int max_exp, const int rounding);
template void BaseRistrettoLayer<float>::Trim2PowerOf2_cpu(float* data,
    const int cnt, const int min_exp, const int max_exp, const int rounding);
template double BaseRistrettoLayer<double>::RandUniform_cpu();
template double BaseRistrettoLayer<float>::RandUniform_cpu();

}  // namespace caffe
