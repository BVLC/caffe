#include "caffe/layers/moe_layer.hpp"
#include "caffe/net.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::LayerSetUp(
                                            const vector<Blob<MItype>*>& bottom,
                                            const vector<Blob<MOtype>*>& top) {
  MOEParameter moe_param = this->layer_param().moe_param();
  if (moe_param.has_gating_net()) {
    gating_net_ = make_shared<Net<float> >(moe_param.gating_net(),
                                           this->device_);
  } else {
    LOG(FATAL) << "MOE Layer requires a gating network.";
  }
  for (size_t i = 0; i < moe_param.expert_net_size(); ++i) {
    expert_nets_.push_back(make_shared<Net<float> >(moe_param.expert_net(i),
                           this->device_));
  }
  if (expert_nets_.size() == 0) {
    LOG(FATAL) << "MOE Layer requires at least one expert network.";
  }
}

/*
template<typename Dtype, typename MItype, typename MOtype>
vector<MOtype> MOELayer::ForwardGatingNetwork(const vector<>) {
  gating_net_()->Forward();
}

template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::ForwardExperts(vector<MOtype> selector) {
  for (size_t i = 0; i < this->expert_nets_.size(); ++i) {
    if (selector[i] > 0) {

    }
  }
}
*/

template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::Reshape(const vector<Blob<MItype>*>& bottom,
                                         const vector<Blob<MOtype>*>& top) {

}

template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::Forward_cpu(
                                        const vector<Blob<MItype>*>& bottom,
                                        const vector<Blob<MOtype>*>& top) {

}


template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::Backward_cpu(
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
