#ifndef CAFFE_BASE_RISTRETTO_LAYER_CUH_
#define CAFFE_BASE_RISTRETTO_LAYER_CUH_

#include <curand_kernel.h>

namespace caffe {


// Returns a random number in (0,1].
// Even though the repetitive initialization of a curand state might look
// suboptimal, the performance is actually nearly the same as when using global
// states.
__device__ __forceinline__ double
RandUniform_device(const int index) {
  curandState state;
  curand_init( (unsigned long long) clock() + index, 0, 0, &state);
  return curand_uniform_double(&state);
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
__device__ void
Trim2MiniFloat_device(Dtype* data, const int bw_mant, const int bw_exp,
      const int rounding, const int index) {
  int bias_out = powf(2, bw_exp - 1) -1;
  float_cast d2;
  // This casts the input to single precision
  d2.d = (float)*data;
  int exponent=d2.parts.exponent - 127 + bias_out;
  double mantisa = d2.parts.mantisa;
  //special case: input is zero or denormalized number
  if (d2.parts.exponent == 0) {
    *data = 0;
    return;
  }
  // Special case: denormalized number as output
  if (exponent < 0) {
    *data = 0;
    return;
  }
  // Saturation: input float is larger than maximum output float
  int max_exp = powf(2, bw_exp) - 1;
  int max_mant = powf(2, bw_mant) - 1;
  if (exponent > max_exp) {
    exponent = max_exp;
    mantisa = max_mant;
  } else {
    // Convert mantissa from long format to short one. Cut off LSBs.
    double tmp = mantisa / powf(2, 23 - bw_mant);
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:{
      mantisa = rint(tmp);
      break;}
    case QuantizationParameter_Rounding_STOCHASTIC:{
      mantisa = __float2int_rd(tmp + RandUniform_device(index));
      break;}
    default:{
      break;}
    }
  }
  // Assemble result
  *data = powf(-1, d2.parts.sign) * ( (mantisa + powf(2, bw_mant)) /
      powf(2, bw_mant) ) * powf(2, exponent - bias_out);
}

}  // namespace caffe

#endif  // CAFFE_BASE_RISTRETTO_LAYER_CUH_
