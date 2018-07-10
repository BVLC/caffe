#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ReLULayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  NeuronLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}


template<typename Dtype,
         typename std::enable_if<unsigned_integer_is_same<Dtype>::value,
         int>::type = 0>
inline void forward(int_tp count, const Dtype* bottom_data, Dtype* top_data,
        Dtype negative_slope, const QuantizerValues* const bottom_qv = nullptr,
        const QuantizerValues* const top_qv = nullptr) {
  typedef typename std::conditional<sizeof(Dtype) == 1, int16_t,
          typename std::conditional<sizeof(Dtype) == 2, int32_t,
                                    int64_t>::type>::type Difftype;
  typedef typename std::conditional<sizeof(Dtype) == 1,
                                    int32_t, int64_t>::type Acctype;
  int8_t shift_bits = (32 / sizeof(Dtype)) - 1;
  Acctype mult;
  int8_t shift;
  QuantizerBase::ScaleQuantVals<Acctype>(bottom_qv, top_qv,
                                         &mult, &shift, shift_bits);
  Difftype bottom_zero = bottom_qv->get_zero<Difftype>();
  Acctype top_zero = top_qv->get_zero<Acctype>();
  Acctype top_min = top_qv->get_min<Acctype>();
  Acctype top_max = top_qv->get_max<Acctype>();
  for (int_tp i = 0; i < count; ++i) {
    Difftype relu = std::max(static_cast<Difftype>(
        static_cast<Difftype>(bottom_data[i]) - bottom_zero), Difftype(0));
    Acctype reg = static_cast<Acctype>((static_cast<int64_t>(relu) *
                   static_cast<int64_t>(mult)) / (1ll << shift_bits));
    if (shift >= 0) {
      reg = reg >> shift;
    } else {
      reg = reg << -shift;
    }
    top_data[i] = static_cast<Dtype>(std::min(std::max(reg + top_zero,
                                                       top_min), top_max));
  }
}

template<typename Dtype,
         typename std::enable_if<float_is_same<Dtype>::value,
         int>::type = 0>
inline void forward(int_tp count, const Dtype* bottom_data, Dtype* top_data,
        Dtype negative_slope, const QuantizerValues* const bottom_qv = nullptr,
        const QuantizerValues* const top_qv = nullptr) {
  for (int_tp i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}


template<typename Dtype, typename MItype, typename MOtype>
void ReLULayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int_tp count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  QuantizerValues bottom_qv = this->bottom_quants_[0]->out_quantizer_values();
  QuantizerValues top_qv = this->top_quants_[0]->in_quantizer_values();
  forward<Dtype>(count, bottom_data, top_data, negative_slope,
                 &bottom_qv, &top_qv);
}


template<typename Dtype, typename MItype, typename MOtype>
void ReLULayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int_tp count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int_tp i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(ReLULayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(ReLULayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(ReLULayer, (double), (double), (double));
INSTANTIATE_CLASS_3T_GUARDED(ReLULayer, (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASS_3T_GUARDED(ReLULayer, (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASS_3T_GUARDED(ReLULayer, (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASS_3T_GUARDED(ReLULayer, (uint64_t), (uint64_t), (uint64_t));

REGISTER_LAYER_CLASS(ReLU);
REGISTER_LAYER_CLASS_INST(ReLU, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(ReLU, (uint8_t), (uint8_t), (uint8_t));
REGISTER_LAYER_CLASS_INST(ReLU, (uint16_t), (uint16_t), (uint16_t));
REGISTER_LAYER_CLASS_INST(ReLU, (uint32_t), (uint32_t), (uint32_t));
REGISTER_LAYER_CLASS_INST(ReLU, (uint64_t), (uint64_t), (uint64_t));

}  // namespace caffe
