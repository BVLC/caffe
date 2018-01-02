#include "caffe/layers/moe_layer.hpp"
#include "caffe/net.hpp"

namespace caffe {


template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::Forward_gpu(
                                        const vector<Blob<MItype>*>& bottom,
                                        const vector<Blob<MOtype>*>& top) {

}


template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::Backward_gpu(
                                        const vector<Blob<MOtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<MItype>*>& bottom) {

}




INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (half_fp), (half_fp),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (float), (float),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (double), (double),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (int8_t), (int8_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (int16_t), (int16_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (int32_t), (int32_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (int64_t), (int64_t),
                             PROTO_TYPES);
}  // namespace caffe
