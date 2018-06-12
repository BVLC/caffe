#include "caffe/layers/moe_layer.hpp"
#include "caffe/net.hpp"

namespace caffe {


template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::Forward_gpu(
                                        const vector<Blob<MItype>*>& bottom,
                                        const vector<Blob<MOtype>*>& top) {
  int_tp select_experts = this->layer_param().moe_param().select_experts();

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
  gating_ = static_cast<Blob<MOtype>*>(this->gating_net_->Forward(&loss)[0]);
  MOtype* gating_data = gating_->mutable_cpu_data();

  vector<int_tp> select_count(gating_->shape()[1], 0);
  vector<vector<int_tp> > batch_selectors;

  // Reset all top blobs
  for (size_t i = 0; i < top.size(); ++i) {
    vptr<MOtype> top_data = top[i]->mutable_gpu_data();
    this->device_->template set<MOtype>(top[i]->count(), MOtype(0), top_data);
  }

  // Select top #select_experts
  for (size_t i = 0; i < gating_->shape()[0]; ++i) {
    vector<int_tp> expert_selectors(select_experts, -1);
    for (size_t j = 0; j < gating_->shape()[1]; ++j) {
      for (size_t k = 0; k < select_experts; ++k) {
        if (expert_selectors[k] == -1 ||
            gating_data[i * gating_->shape()[1] + expert_selectors[k]] <
            gating_data[i * gating_->shape()[1] + j]) {
          for (size_t l = select_experts-1; l > k; --l) {
            expert_selectors[l] = expert_selectors[l - 1];
          }
          expert_selectors[k] = j;
          break;
        }
      }
    }
    for(size_t k = 0; k < select_experts; ++k) {
      select_count[expert_selectors[k]] += 1;
    }
    batch_selectors.push_back(expert_selectors);
  }

  // Generate load balancing loss
  if (this->phase_ == caffe::TRAIN) {
    MOtype* observed_count = top[top.size()-2]->mutable_cpu_data();
    MOtype* expected_count = top[top.size()-1]->mutable_cpu_data();
    for (size_t j = 0; j < gating_->shape()[1]; ++j) {
      MOtype norm_observed = static_cast<MOtype>(select_count[j])
          / static_cast<MOtype>(gating_->shape()[0]);
      MOtype norm_expected = static_cast<MOtype>(select_experts)
          / static_cast<MOtype>(gating_->shape()[1]);
      for (size_t i = 0; i < gating_->shape()[0]; ++i) {
        observed_count[i * select_count.size() + j] = norm_observed;
        expected_count[i * select_count.size() + j] = norm_expected;
      }
    }
  }

  // Make gating data sparse and renormalize
  for (size_t i = 0; i < gating_->shape()[0]; ++i) {
    MOtype norm = MOtype(0);
    for (size_t j = 0; j < gating_->shape()[1]; ++j) {
      MOtype select = MOtype(0);
      for (size_t k = 0; k < select_experts; ++k) {
        if (batch_selectors[i][k] == j) {
          // std::cout << "Select " << select_experts << ", " << i << ", "
          //           << k << ", " << j << std::endl;
          select = MOtype(1);
          break;
        }
      }
      gating_data[i * gating_->shape()[1] + j] *= select;
      norm += gating_data[i * gating_->shape()[1] + j];
    }
    for (size_t j = 0; j < gating_->shape()[1]; ++j) {
      gating_data[i * gating_->shape()[1] + j] /= norm;
    }
  }

  // Forward experts
  if (this->phase_ == caffe::TEST &&
      !this->layer_param().moe_param().full_forward()) {
    // Forward expert networks (partial, only forward selected experts
    // per batch item)
#pragma omp parallel for num_threads(this->parallel_nets_)
    for (size_t i = 0; i < gating_->shape()[0]; ++i) {
      Caffe::SelectDevice(this->device_->id(), false);
#ifdef USE_OPENMP
      int_tp tidx = omp_get_thread_num();
#else  // USE_OPENMP
      int_tp tidx = 0;
#endif  // USE_OPENMP
      this->device_->SwitchQueue(i);

      vector<int_tp> expert_selectors = batch_selectors[i];
      for (size_t p = 0; p < select_experts; ++p) {
        const vector<BlobBase*>& expert_input_blobs =
                   this->expert_nets_[expert_selectors[p]][tidx]->input_blobs();
        int_tp k = 0;
        for (size_t l = 0; l < bottom.size(); ++l) {
          if (this->layer_param().moe_param().map_bottom_size() <= l ||
              this->layer_param().moe_param().map_bottom(l) ==
                  MOEParameter_BottomMapping_EXPERT ||
              this->layer_param().moe_param().map_bottom(l) ==
                  MOEParameter_BottomMapping_GATING_AND_EXPERT) {
            this->device_->template copy<MItype>(bottom[l]->count(1),
                               bottom[l]->gpu_data() + i * bottom[l]->count(1),
                               static_cast<Blob<MItype>*>(
                                    expert_input_blobs[k])->mutable_gpu_data());
            ++k;
          }
        }
        const vector<BlobBase*> result_vec =
                     this->expert_nets_[expert_selectors[p]][tidx]->
                                                                 Forward(&loss);
        for (size_t k = 0; k < result_vec.size(); ++k) {
          Blob<MOtype>* result = static_cast<Blob<MOtype>*>(result_vec[k]);
          this->device_->template axpby<MOtype>(
              top[k]->count(1),
              gating_data[i * gating_->shape()[1] + expert_selectors[p]],
              result->gpu_data(),
              MOtype(1),
              top[k]->mutable_gpu_data() + i * top[k]->count(1));
        }
      }
    }
    this->device_->FinishQueues();
  } else {
    // Forward expert networks (full batch)
    for (size_t j = 0; j < this->expert_nets_.size(); ++j) {
      const vector<BlobBase*>& expert_input_blobs = this->expert_nets_[j][0]->
          input_blobs();
      int_tp k = 0;
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
      const vector<BlobBase*> result_vec =
                                       this->expert_nets_[j][0]->Forward(&loss);
      for (size_t k = 0; k < result_vec.size(); ++k) {
        Blob<MOtype>* result = static_cast<Blob<MOtype>*>(result_vec[k]);
        for (size_t i = 0; i < gating_->shape()[0]; ++i) {
          this->device_->template axpby(
                      top[k]->count(1),
                      gating_data[i * gating_->shape()[1] + j],
                      result->gpu_data() + i * top[k]->count(1),
                      MOtype(1),
                      top[k]->mutable_gpu_data() + i * top[k]->count(1));
        }
      }
    }
  }
}


template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::Backward_gpu(
                                        const vector<Blob<MOtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<MItype>*>& bottom) {

  // Reset all bottom blob diffs
  for (size_t i = 0; i < bottom.size(); ++i) {
    MItype* bottom_diff = bottom[i]->mutable_cpu_diff();
    caffe_set(bottom[i]->count(), MItype(0), bottom_diff);
  }

  // Set gating diff to load balancing diff
  const MOtype* gating_data = gating_->cpu_data();
  MOtype* gating_diff = gating_->mutable_cpu_diff();
  const MOtype* observed_diff = top[top.size()-2]->cpu_diff();
  caffe_copy(gating_->count(), observed_diff, gating_diff);

  // Backward all experts
  for (size_t j = 0; j < this->expert_nets_.size(); ++j) {
    const vector<BlobBase*>& expert_output_blobs = this->expert_nets_[j][0]->
                                                                 output_blobs();
    for (size_t k = 0; k < expert_output_blobs.size(); ++k) {
      for (size_t i = 0; i < gating_->shape()[0]; ++i) {
        // Compute diff w.r.t expert outputs
        this->device_->template scale<MOtype>(top[k]->count(1),
                    gating_data[i * gating_->shape()[1] + j],
                    top[k]->gpu_diff() + i * top[k]->count(1),
                    static_cast<Blob<MOtype>*>(expert_output_blobs[k])->
                                     mutable_gpu_diff() + i * top[k]->count(1));
        // Compute diff w.r.t gating outputs
        gating_diff[i * gating_->shape()[1] + j] += caffe_dot(top[k]->count(1),
                            top[k]->cpu_diff() + i * top[k]->count(1),
                            static_cast<Blob<MOtype>*>(expert_output_blobs[k])->
                              cpu_data() + i * top[k]->count(1));
      }
    }

    // Backward expert networks (full)
    this->expert_nets_[j][0]->Backward();
    const vector<BlobBase*>& expert_input_blobs = this->expert_nets_[j][0]->
                                                                 input_blobs();
    int_tp k = 0;
    for (size_t l = 0; l < bottom.size(); ++l) {
      if (this->layer_param().moe_param().map_bottom_size() <= l ||
          this->layer_param().moe_param().map_bottom(l) ==
              MOEParameter_BottomMapping_EXPERT ||
          this->layer_param().moe_param().map_bottom(l) ==
              MOEParameter_BottomMapping_GATING_AND_EXPERT) {
        if (propagate_down[l]) {
          this->device_->template axpby<MItype>(
              bottom[l]->count(),
              MItype(1),
              static_cast<Blob<MItype>*>(expert_input_blobs[k])->gpu_diff(),
              MItype(1),
              bottom[l]->mutable_gpu_diff());
        }
        ++k;
      }
    }
  }

  // Backward gating network
  this->gating_net_->Backward();
  const vector<BlobBase*>& gating_input_blobs = this->gating_net_->
                                                                  input_blobs();
  size_t k = 0;
  for (size_t l = 0; l < bottom.size(); ++l) {
    if (this->layer_param().moe_param().map_bottom_size() <= l ||
        this->layer_param().moe_param().map_bottom(l) ==
            MOEParameter_BottomMapping_GATING ||
        this->layer_param().moe_param().map_bottom(l) ==
            MOEParameter_BottomMapping_GATING_AND_EXPERT) {
      if (propagate_down[l]) {
        this->device_->template axpby<MItype>(
            bottom[l]->count(),
            MItype(1),
            static_cast<Blob<MItype>*>(gating_input_blobs[k])->gpu_diff(),
            MItype(1),
            bottom[l]->mutable_gpu_diff());
      }
      ++k;
    }
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Forward_gpu,
                                 (half_fp), (half_fp), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Forward_gpu,
                                 (float), (float), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Forward_gpu,
                                 (double), (double), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Forward_gpu,
                                  (uint8_t), (uint8_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Forward_gpu,
                                  (uint16_t), (uint16_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Forward_gpu,
                                  (uint32_t), (uint32_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Forward_gpu,
                                  (uint64_t), (uint64_t),  PROTO_TYPES);


INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Backward_gpu,
                                 (half_fp), (half_fp), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Backward_gpu,
                                 (float), (float), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Backward_gpu,
                                 (double), (double), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Backward_gpu,
                                  (uint8_t), (uint8_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Backward_gpu,
                                  (uint16_t), (uint16_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Backward_gpu,
                                  (uint32_t), (uint32_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MOELayer, Backward_gpu,
                                  (uint64_t), (uint64_t),  PROTO_TYPES);
}  // namespace caffe
