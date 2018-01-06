#include "caffe/layers/moe_layer.hpp"
#include "caffe/net.hpp"

namespace caffe {


template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::Forward_gpu(
                                        const vector<Blob<MItype>*>& bottom,
                                        const vector<Blob<MOtype>*>& top) {
  // Preload gating network blobs
  const vector<BlobBase*>& gating_input_blobs = this->gating_net_->
      input_blobs();
  size_t k = 0;
  for (size_t l = 0; l < bottom.size(); ++l) {
    if (this->layer_param().moe_param().map_bottom_size() <= l ||
        this->layer_param().moe_param().map_bottom(l) ==
            MOEParameter_BottomMapping_GATING ||
        this->layer_param().moe_param().map_bottom(l) ==
            MOEParameter_BottomMapping_GATING_AND_EXPERT) {
      this->device_->template copy<MItype>(bottom[l]->count(),
         bottom[l]->gpu_data(),
         static_cast<Blob<MItype>*>(gating_input_blobs[k])->mutable_gpu_data());
      ++k;
    }
  }
  // Forward gating network
  float loss = 0;
  const Blob<MOtype>* gating = static_cast<Blob<MOtype>*>(
      this->gating_net_->Forward(&loss)[0]);

  vector<vector<Blob<MOtype>*> > experts_result_vec(this->expert_nets_.size());
  loss = 0;
  const MOtype* gating_data = gating->cpu_data();
  MOtype eps = MOtype(0);
  size_t j = 0;
#pragma omp parallel for
  for (size_t i = 0; i < this->expert_nets_.size(); ++i) {
    vector<Blob<MOtype>*> result_vec;
    // If the gating network selects this expert, preload blobs and forward
    if (gating_data[i] > eps) {
      const vector<BlobBase*>& expert_input_blobs = this->expert_nets_[i]->
          input_blobs();
      size_t k = 0;
      for (size_t l = 0; l < bottom.size(); ++l) {
        if (this->layer_param().moe_param().map_bottom_size() <= l ||
            this->layer_param().moe_param().map_bottom(l) ==
                MOEParameter_BottomMapping_EXPERT ||
            this->layer_param().moe_param().map_bottom(l) ==
                MOEParameter_BottomMapping_GATING_AND_EXPERT) {
          this->device_->template copy<MItype>(bottom[l]->count(),
             bottom[l]->gpu_data(),
             static_cast<Blob<MItype>*>(expert_input_blobs[k])->
             mutable_gpu_data());
          ++k;
        }
      }
      // Forward expert network
      const vector<BlobBase*> result = this->expert_nets_[i]->Forward(&loss);
      for (size_t k = 0; k < result.size(); ++k) {
        result_vec[k] = (static_cast<Blob<MOtype>*>(result[k]));
      }
      ++j;
    }
    experts_result_vec[i] = result_vec;
  }

  // Loop over all top blobs
#pragma omp parallel for
  for (size_t i = 0; i < top.size(); ++i) {
   vptr<MOtype> top_data = top[i]->mutable_gpu_data();
    this->device_->template set<MOtype>(top[i]->count(), MOtype(0), top_data);
    // Loop over all experts
    for (size_t j = 0; j < experts_result_vec.size(); ++j) {
      // Only consider experts that have been selected (sparse)
      if (experts_result_vec[j].size() > i) {
        vptr<const MOtype> expert_data = experts_result_vec[j][i]->gpu_data();
        this->device_->axpby(top[i]->count(), gating_data[j], expert_data,
                             MOtype(1), top_data);
      }
    }
  }
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
