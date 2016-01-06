#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

#ifndef CPU_ONLY
template <typename Dtype>
void rmsprop_update_gpu(int N, Dtype* g, Dtype* h, Dtype rms_decay,
    Dtype delta, Dtype local_rate);
#endif

template <typename Dtype>
void RMSPropSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();

  // get the learning rate
  Dtype delta = this->param_.delta();
  Dtype rms_decay = this->param_.rms_decay();
  Dtype local_rate = rate * net_params_lr[param_id];

  switch (Caffe::mode()) {
  case Caffe::CPU:
    // compute square of gradient in update
    caffe_powx(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_cpu_data());

    // update history
    caffe_cpu_axpby(net_params[param_id] -> count(),
        Dtype(1-rms_decay), this->update_[param_id]->cpu_data(),
        rms_decay, this->history_[param_id]-> mutable_cpu_data());

    // prepare update
    caffe_powx(net_params[param_id]->count(),
        this->history_[param_id]->cpu_data(), Dtype(0.5),
        this->update_[param_id]->mutable_cpu_data());

    caffe_add_scalar(net_params[param_id]->count(),
        delta, this->update_[param_id]->mutable_cpu_data());

    caffe_div(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(), this->update_[param_id]->cpu_data(),
        this->update_[param_id]->mutable_cpu_data());

    // scale and copy
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
        this->update_[param_id]->cpu_data(), Dtype(0),
        net_params[param_id]->mutable_cpu_diff());
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    rmsprop_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        this->history_[param_id]->mutable_gpu_data(),
        rms_decay, delta, local_rate);
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(RMSPropSolver);
REGISTER_SOLVER_CLASS(RMSProp);

}  // namespace caffe
