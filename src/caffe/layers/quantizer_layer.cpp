#include "caffe/layers/quantizer_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void QuantizerLayer<Dtype, MItype, MOtype>::LayerSetUp(
      const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  this->Reshape(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void QuantizerLayer<Dtype, MItype, MOtype>::Reshape(
      const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size())
      << "Expected equal number of top and bottom blobs.";
  for (size_t i = 0; i < bottom.size(); ++i) {
    top[i]->ReshapeLike(bottom[i]);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void QuantizerLayer<Dtype, MItype, MOtype>::Forward_cpu(
      const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  for(size_t i = 0; i < bottom.size(); ++i) {
    this->top_quant_->Forward_cpu(bottom[i], top[i], true, false);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void QuantizerLayer<Dtype, MItype, MOtype>::Backward_cpu(
      const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom) {
  for(size_t i = 0; i < top.size(); ++i) {
    if (propagate_down[i]) {
      this->top_quant_->Backward_cpu(top[i], bottom[i], false, true);
    }
  }
}

#ifndef CPU_ONLY
template<typename Dtype, typename MItype, typename MOtype>
void QuantizerLayer<Dtype, MItype, MOtype>::Forward_gpu(
      const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  for(size_t i = 0; i < bottom.size(); ++i) {
    this->top_quant_->Forward_gpu(bottom[i], top[i], true, false);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void QuantizerLayer<Dtype, MItype, MOtype>::Backward_gpu(
      const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom) {
  for(size_t i = 0; i < top.size(); ++i) {
    if (propagate_down[i]) {
      this->top_quant_->Backward_gpu(top[i], bottom[i], false, true);
    }
  }
}

#else  // CPU_ONLY
STUB_GPU(QuantizerLayer);
#endif  // CPU_ONLY

INSTANTIATE_CLASS_3T_GUARDED(QuantizerLayer, (half_fp), (half_fp), PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(QuantizerLayer, (float), (float), PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(QuantizerLayer, (double), (double), PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(QuantizerLayer, (int8_t), (int8_t), PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(QuantizerLayer, (int16_t), (int16_t), PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(QuantizerLayer, (int32_t), (int32_t), PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(QuantizerLayer, (int64_t), (int64_t), PROTO_TYPES);

REGISTER_LAYER_CLASS(Quantizer);
REGISTER_LAYER_CLASS_INST(Quantizer, (half_fp), (half_fp), PROTO_TYPES);
REGISTER_LAYER_CLASS_INST(Quantizer, (float), (float), PROTO_TYPES);
REGISTER_LAYER_CLASS_INST(Quantizer, (double), (double), PROTO_TYPES);
REGISTER_LAYER_CLASS_INST(Quantizer, (int8_t), (int8_t), PROTO_TYPES);
REGISTER_LAYER_CLASS_INST(Quantizer, (int16_t), (int16_t), PROTO_TYPES);
REGISTER_LAYER_CLASS_INST(Quantizer, (int32_t), (int32_t), PROTO_TYPES);
REGISTER_LAYER_CLASS_INST(Quantizer, (int64_t), (int64_t), PROTO_TYPES);

}
