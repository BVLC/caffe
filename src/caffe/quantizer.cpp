#include "caffe/quantizer.hpp"

namespace caffe {

QuantizerBase::QuantizerBase(QuantizerParameter& param)
  : quant_param_(param) {
  device_ = Caffe::GetDevice(quant_param_.device(), true);
}

template<typename MItype, typename MOtype>
Quantizer<MItype, MOtype>::Quantizer(QuantizerParameter& param)
  : QuantizerBase(param) {

}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_cpu(
   size_t n, const void* input, void* output) {
  this->Forward_cpu(n,
                    static_cast<const MItype*>(input),
                    static_cast<MOtype*>(output));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_cpu(
   size_t n, const MItype* input, MOtype* output) {
  for (size_t i = 0; i < n; ++i) {
    output[i] = static_cast<MOtype>(input[i]);
  }
}


INSTANTIATE_CLASS_2T(Quantizer,
                     (half_fp)(float)(double),
                     (half_fp)(float)(double))


}  // namespace caffe
