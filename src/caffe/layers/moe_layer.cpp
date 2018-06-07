#include "caffe/layers/moe_layer.hpp"
#include "caffe/net.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
vector<shared_ptr<QuantizerBase> >
                         MOELayer<Dtype, MItype, MOtype>::get_all_quantizers() {
  vector<shared_ptr<QuantizerBase> > quants =
      Layer<Dtype, MItype, MOtype>::get_all_quantizers();

  // Add all quantizers from the gating net
  vector<shared_ptr<QuantizerBase> > gating_quants =
      gating_net_->get_all_quantizers();
  for (size_t l = 0; l < gating_quants.size(); ++l) {
    quants.push_back(gating_quants[l]);
  }

  // Add all quantizers from the expert nets
  for (size_t j = 0; j < expert_nets_.size(); ++j) {
    for (size_t k = 0; k < expert_nets_[j].size(); ++k) {
      vector<shared_ptr<QuantizerBase> > expert_quants =
          expert_nets_[j][k]->get_all_quantizers();
      for (size_t l = 0; l < expert_quants.size(); ++l) {
        quants.push_back(expert_quants[l]);
      }
    }
  }

  return quants;
}


template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::LayerSetUp(
                                            const vector<Blob<MItype>*>& bottom,
                                            const vector<Blob<MOtype>*>& top) {
  MOEParameter moe_param = this->layer_param().moe_param();
  if (moe_param.has_gating_net()) {
    gating_net_ = make_shared<Net<float> >(moe_param.gating_net(),
                                           this->device_);
    vector<shared_ptr<BlobBase> > gating_net_params = gating_net_->params();
    for (size_t i = 0; i < gating_net_params.size(); ++i) {
      this->blobs_.push_back(
          std::static_pointer_cast<Blob<Dtype> >(gating_net_params[i]));
    }
  } else {
    LOG(FATAL) << "MOE Layer requires a gating network.";
  }
  for (size_t i = 0; i < moe_param.expert_net_size(); ++i) {
    size_t instances = 1;
    if (moe_param.expert_instances_size() > i) {
      instances = moe_param.expert_instances(i);
    }
    for (size_t j = 0; j < instances; ++j) {
      vector<shared_ptr<Net<float> > > expert_nets;
      vector<shared_ptr<BlobBase> > expert_net_params_zero;
      for (size_t k = 0;
           k < (Caffe::mode() == TEST ? bottom[0]->shape()[0] : 1); ++k) {
        shared_ptr<Net<float> > expert_net =
            make_shared<Net<float> >(moe_param.expert_net(i), this->device_);
        vector<shared_ptr<BlobBase> > expert_net_params = expert_net->params();
        if (k == 0) {
          // If multiple copies of an expert exists, register the first and
          // copy to the others (shared parameters)
          for (size_t i = 0; i < expert_net_params.size(); ++i) {
            this->blobs_.push_back(
                std::static_pointer_cast<Blob<Dtype> >(expert_net_params[i]));
          }
          expert_net_params_zero = expert_net_params;
        } else {
          for (size_t i = 0; i < expert_net_params.size(); ++i) {
            expert_net_params[i]->ShareDataBase(
                                             expert_net_params_zero[i].get());
            expert_net_params[i]->ShareDiffBase(
                                             expert_net_params_zero[i].get());
          }
        }
        expert_nets.push_back(expert_net);
      }
      expert_nets_.push_back(expert_nets);
    }
  }
  if (expert_nets_.size() == 0) {
    LOG(FATAL) << "MOE Layer requires at least one expert network.";
  }
  this->InitializeQuantizers(bottom, top);
}


template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::Reshape(
                                         const vector<Blob<MItype>*>& bottom,
                                         const vector<Blob<MOtype>*>& top) {
  const vector<BlobBase*>& gating_input_blobs = this->gating_net_->
      input_blobs();
  size_t k = 0;
  for (size_t l = 0; l < bottom.size(); ++l) {
    if (this->layer_param().moe_param().map_bottom_size() <= l ||
        this->layer_param().moe_param().map_bottom(l) ==
            MOEParameter_BottomMapping_GATING ||
        this->layer_param().moe_param().map_bottom(l) ==
            MOEParameter_BottomMapping_GATING_AND_EXPERT) {
      gating_input_blobs[k]->Reshape(bottom[l]->shape());
      ++k;
    }
  }
  this->gating_net_->Reshape();

  for (size_t j = 0; j < expert_nets_.size(); ++j) {
    for (size_t i = 0; i < expert_nets_[j].size(); ++i) {
      const vector<BlobBase*>& expert_input_blobs = this->expert_nets_[j][i]->
          input_blobs();
      size_t k = 0;
      for (size_t l = 0; l < bottom.size(); ++l) {
        if (this->layer_param().moe_param().map_bottom_size() <= l ||
            this->layer_param().moe_param().map_bottom(l) ==
                MOEParameter_BottomMapping_EXPERT ||
            this->layer_param().moe_param().map_bottom(l) ==
                MOEParameter_BottomMapping_GATING_AND_EXPERT) {
          vector<int_tp> shape = bottom[l]->shape();
          shape[0] = Caffe::mode() == TEST ? 1 : shape[0];
          expert_input_blobs[k]->Reshape(shape);
          ++k;
        }
      }
      this->expert_nets_[j][i]->Reshape();
      if (j == 0 and i == 0) {
        const vector<BlobBase*>& expert_output_blobs =
            this->expert_nets_[j][i]->output_blobs();
        for (size_t l = 0; l < top.size(); ++l) {
          top[l]->Reshape(expert_output_blobs[l]->shape());
        }
      }
    }
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::Forward_cpu(
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
      caffe_copy<MItype>(bottom[l]->count(), bottom[l]->cpu_data(),
         static_cast<Blob<MItype>*>(gating_input_blobs[k])->mutable_cpu_data());
      ++k;
    }
  }

  // Forward gating network
  float loss = 0;
  gating_ = static_cast<Blob<MOtype>*>(
      this->gating_net_->Forward(&loss)[0]);
  MOtype* gating_data = gating_->mutable_cpu_data();

  vector<vector<int_tp> > batch_selectors;

  // Reset all top blobs
  for (size_t i = 0; i < top.size(); ++i) {
    MOtype* top_data = top[i]->mutable_cpu_data();
    caffe_set(top[i]->count(), MOtype(0), top_data);
  }

  // Select top #select_experts
  for (size_t i = 0; i < gating_->shape()[0]; ++i) {
    vector<int_tp> expert_selectors(select_experts, -1);
    for (size_t j = 0; j < gating_->shape()[1]; ++j) {
      for (size_t k = 0; k < select_experts; ++k) {
        if (expert_selectors[k] == -1 ||
            gating_data[i * gating_->shape()[1] + expert_selectors[k]]
                        < gating_data[i * gating_->shape()[1] + j]) {
          for (size_t l = k + 1; l < select_experts; ++l) {
            expert_selectors[l] = expert_selectors[l - 1];
          }
          expert_selectors[k] = j;
          break;
        }
      }
    }
    batch_selectors.push_back(expert_selectors);
  }

  // Make gating data sparse and renormalize
  for (size_t i = 0; i < gating_->shape()[0]; ++i) {
    MOtype norm = MOtype(0);
    for (size_t j = 0; j < gating_->shape()[1]; ++j) {
      MOtype select = MOtype(0);
      for (size_t k = 0; k < select_experts; ++k) {
        if (batch_selectors[i][k] == j) {
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
  if (Caffe::mode() == TEST) {
    // Forward expert networks (partial, only forward selected experts
    // per batch item)
    for (size_t i = 0; i < gating_->shape()[0]; ++i) {
      vector<int_tp> expert_selectors = batch_selectors[i];
      for (size_t p = 0; p < select_experts; ++p) {
        const vector<BlobBase*>& expert_input_blobs =
            this->expert_nets_[expert_selectors[p]][i]->input_blobs();
        int_tp k = 0;
        for (size_t l = 0; l < bottom.size(); ++l) {
          if (this->layer_param().moe_param().map_bottom_size() <= l ||
              this->layer_param().moe_param().map_bottom(l) ==
                  MOEParameter_BottomMapping_EXPERT ||
              this->layer_param().moe_param().map_bottom(l) ==
                  MOEParameter_BottomMapping_GATING_AND_EXPERT) {
            caffe_copy<MItype>(bottom[l]->count(1),
                               bottom[l]->cpu_data() + i * bottom[l]->count(1),
                               static_cast<Blob<MItype>*>(
                                   expert_input_blobs[k])->mutable_cpu_data());
            ++k;
          }
        }
        const vector<BlobBase*> result_vec =
                     this->expert_nets_[expert_selectors[p]][i]->Forward(&loss);
        for (size_t k = 0; k < result_vec.size(); ++k) {
          Blob<MOtype>* result = static_cast<Blob<MOtype>*>(result_vec[k]);
          caffe_axpby(
              top[k]->count(1),
              gating_data[i * gating_->shape()[1] + expert_selectors[p]],
              result->cpu_data() + i * top[k]->count(1),
              MOtype(1),
              top[k]->mutable_cpu_data() + i * top[k]->count(1));
        }
      }
    }
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
          caffe_copy<MItype>(bottom[l]->count(), bottom[l]->cpu_data(),
             static_cast<Blob<MItype>*>(expert_input_blobs[k])->
               mutable_cpu_data());
          ++k;
        }
      }
      const vector<BlobBase*> result_vec =
                                       this->expert_nets_[j][0]->Forward(&loss);
      for (size_t k = 0; k < result_vec.size(); ++k) {
        Blob<MOtype>* result = static_cast<Blob<MOtype>*>(result_vec[k]);
        for (size_t i = 0; i < gating_->shape()[0]; ++i) {
          caffe_axpby(top[k]->count(1),
                      gating_data[i * gating_->shape()[1] + j],
                      result->cpu_data() + i * top[k]->count(1),
                      MOtype(1),
                      top[k]->mutable_cpu_data() + i * top[k]->count(1));
        }
      }
    }
  }
}


template<typename Dtype, typename MItype, typename MOtype>
void MOELayer<Dtype, MItype, MOtype>::Backward_cpu(
                                        const vector<Blob<MOtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<MItype>*>& bottom) {

  // Reset all bottom blob diffs
  for (size_t i = 0; i < bottom.size(); ++i) {
    MItype* bottom_diff = bottom[i]->mutable_cpu_diff();
    caffe_set(bottom[i]->count(), MItype(0), bottom_diff);
  }

  // Reset gating diff
  const MOtype* gating_data = gating_->cpu_data();
  MOtype* gating_diff = gating_->mutable_cpu_diff();
  caffe_set(gating_->count(), MOtype(0), gating_diff);


  // Backward all experts
  for (size_t j = 0; j < this->expert_nets_.size(); ++j) {
    const vector<BlobBase*>& expert_output_blobs = this->expert_nets_[j][0]->
                                                                 output_blobs();
    for (size_t k = 0; k < expert_output_blobs.size(); ++k) {
      for (size_t i = 0; i < gating_->shape()[0]; ++i) {
        // Compute diff w.r.t expert outputs
        caffe_scale(top[k]->count(1),
                    gating_data[i * gating_->shape()[1] + j],
                    top[k]->cpu_diff() + i * top[k]->count(1),
                    static_cast<Blob<MOtype>*>(expert_output_blobs[k])->
                                     mutable_cpu_diff() + i * top[k]->count(1));
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
          caffe_axpby(
              bottom[l]->count(),
              MItype(1),
              static_cast<Blob<MItype>*>(expert_input_blobs[k])->cpu_diff(),
              MItype(1),
              bottom[l]->mutable_cpu_diff());
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
        caffe_axpby(
            bottom[l]->count(),
            MItype(1),
            static_cast<Blob<MItype>*>(gating_input_blobs[k])->cpu_diff(),
            MItype(1),
            bottom[l]->mutable_cpu_diff());
      }
      ++k;
    }
  }
}


INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (half_fp), (half_fp), PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (float), (float), PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (double), (double), PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (uint8_t), (uint8_t), PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (uint16_t), (uint16_t), PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (uint32_t), (uint32_t), PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(MOELayer, (uint64_t), (uint64_t), PROTO_TYPES);
}  // namespace caffe
