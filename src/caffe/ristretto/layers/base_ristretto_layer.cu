#include "ristretto/base_ristretto_layer.hpp"
#include "ristretto/base_ristretto_layer.cuh"

namespace caffe {

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeWeights_gpu(Dtype* weight,
      const int cnt_weight, Dtype* bias, const int cnt_bias,
      const int rounding) {
  switch (this->precision_) {
  case QuantizationParameter_Precision_MINI_FLOATING_POINT:
    this->Trim2FloatingPoint_gpu(weight, cnt_weight, this->fp_mant_,
        this->fp_exp_, rounding);
    this->Trim2FloatingPoint_gpu(bias, cnt_bias, this->fp_mant_, this->fp_exp_,
        rounding);
    break;
  case QuantizationParameter_Precision_FIXED_POINT:
    this->Trim2FixedPoint_gpu(weight, cnt_weight,
        this->bw_params_, rounding, this->fl_params_);
    this->Trim2FixedPoint_gpu(bias, cnt_bias,
        this->bw_params_, rounding, this->fl_params_);
    break;
  case QuantizationParameter_Precision_POWER_2_WEIGHTS:
    this->Trim2PowerOf2_gpu(weight, cnt_weight,
        this->pow_2_min_exp_, this->pow_2_max_exp_, rounding);
    // Don't trim bias
    break;
  default:
    LOG(FATAL) << "Unknown trimming mode: " << this->precision_;
    break;
  }
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeLayerOutputs_gpu(
    Dtype* top_data, const int top_count) {
  switch (this->precision_) {
    case QuantizationParameter_Precision_POWER_2_WEIGHTS:
      break;
    case QuantizationParameter_Precision_FIXED_POINT:
      this->Trim2FixedPoint_gpu(top_data, top_count,
          this->bw_layer_out_, this->rounding_, this->fl_layer_out_);
      break;
    case QuantizationParameter_Precision_MINI_FLOATING_POINT:
      this->Trim2FloatingPoint_gpu(top_data, top_count,
          this->fp_mant_, this->fp_exp_, this->rounding_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << this->precision_;
      break;
  }
}

template <typename Dtype>
__global__ void Trim2FixedPoint_kernel(Dtype* data, const int cnt,
      const int bit_width, const int rounding, const int fl) {
	CUDA_KERNEL_LOOP(index, cnt) {
    // Saturate data
    Dtype max_data = (powf(2, bit_width - 1) - 1) * powf(2, -fl);
    Dtype min_data = -powf(2, bit_width - 1) * powf(2, -fl);
    data[index] = fmax(fmin(data[index], max_data), min_data);
    // Round data
    data[index] /= powf(2, -fl);
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = rint(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = __float2int_rd(data[index] + RandUniform_device(index));
      break;
    default:
      break;
    }
    data[index] *= powf(2, -fl);
	}
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FixedPoint_gpu(Dtype* data, const int cnt,
      const int bit_width, const int rounding, int fl){
  Trim2FixedPoint_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
      data, cnt, bit_width, rounding, fl);
}

template <typename Dtype>
__global__ void Trim2FloatingPoint_kernel(Dtype* data, const int cnt,
      const int bw_mant, const int bw_exp, const int rounding){
	CUDA_KERNEL_LOOP(index, cnt) {
    Trim2FloatingPoint_device(&data[index], bw_mant, bw_exp, rounding, index);
	}
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FloatingPoint_gpu(Dtype* data,
      const int cnt, const int bw_mant, const int bw_exp, const int rounding){
  Trim2FloatingPoint_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
      data, cnt, bw_mant, bw_exp, rounding);
}

template <typename Dtype>
__global__ void Trim2PowerOf2_kernel(Dtype* data, const int cnt,
      const int min_exp, const int max_exp, const int rounding) {
	CUDA_KERNEL_LOOP(index, cnt) {
    float exponent = log2f(fabs((float)data[index]));
    int sign = data[index] >= 0 ? 1 : -1;
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      exponent = rint(exponent);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      exponent = __float2int_rd(exponent + RandUniform_device(index));
      break;
    default:
      break;
    }
    exponent = fmaxf(fminf(exponent, max_exp), min_exp);
    data[index] = sign * powf(2, exponent);
	}
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2PowerOf2_gpu(Dtype* data, const int cnt,
      const int min_exp, const int max_exp, const int rounding) {
  Trim2PowerOf2_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
      data, cnt, min_exp, max_exp, rounding);
}

// Explicit instantiations
template void BaseRistrettoLayer<double>::QuantizeWeights_gpu(double* weight,
    const int cnt_weight, double* bias, const int cnt_bias, const int rounding);
template void BaseRistrettoLayer<float>::QuantizeWeights_gpu(float* weight,
    const int cnt_weight, float* bias, const int cnt_bias, const int rounding);
template void BaseRistrettoLayer<double>::QuantizeLayerOutputs_gpu(
    double* top_data, const int top_count);
template void BaseRistrettoLayer<float>::QuantizeLayerOutputs_gpu(
    float* top_data, const int top_count);
template void BaseRistrettoLayer<double>::Trim2FixedPoint_gpu(double* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseRistrettoLayer<float>::Trim2FixedPoint_gpu(float* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseRistrettoLayer<double>::Trim2FloatingPoint_gpu(double* data,
    const int cnt, const int bw_mant, const int bw_exp, const int rounding);
template void BaseRistrettoLayer<float>::Trim2FloatingPoint_gpu(float* data,
    const int cnt, const int bw_mant, const int bw_exp, const int rounding);
template void BaseRistrettoLayer<double>::Trim2PowerOf2_gpu(double* data,
    const int cnt, const int min_exp, const int max_exp, const int rounding);
template void BaseRistrettoLayer<float>::Trim2PowerOf2_gpu(float* data,
    const int cnt, const int min_exp, const int max_exp, const int rounding);

}  // namespace caffe


