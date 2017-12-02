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
void Quantizer<MItype, MOtype>::Forward_cpu(Blob<MItype>* input,
                                            Blob<MOtype>* output,
                                            bool fw_data, bool fw_diff) {
  CHECK_EQ(input->count(), output->count());
  if (fw_data) {
    this->Forward_cpu(input->count(),
                      input->cpu_data(),
                      output->mutable_cpu_data());
  }
  if (fw_diff) {
    this->Forward_cpu(input->count(),
                      input->cpu_diff(),
                      output->mutable_cpu_diff());
  }
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
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    output[i] = static_cast<MOtype>(input[i]);
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_cpu(Blob<MOtype>* input,
                                             Blob<MItype>* output,
                                             bool bw_data, bool bw_diff) {
  CHECK_EQ(input->count(), output->count());
  if (bw_data) {
    this->Backward_cpu(input->count(),
                       input->cpu_data(),
                       output->mutable_cpu_data());
  }
  if (bw_diff) {
    this->Backward_cpu(input->count(),
                       input->cpu_diff(),
                       output->mutable_cpu_diff());
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_cpu(
   size_t n, const void* input, void* output) {
  this->Backward_cpu(n,
                    static_cast<const MOtype*>(input),
                    static_cast<MItype*>(output));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_cpu(
   size_t n, const MOtype* input, MItype* output) {
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    output[i] = static_cast<MItype>(input[i]);
  }
}


INSTANTIATE_CLASS_2T(Quantizer, VARIANT_TYPES, VARIANT_TYPES)


}  // namespace caffe
