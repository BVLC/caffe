#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void AdaDeltaSolver<Dtype>::AdaDeltaPreSolve() {
  // Add the extra history entries for AdaDelta after those from
  // SGDSolver::PreSolve
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
        const vector<int>& shape = net_params[i]->shape();
        this->history_.push_back(
                shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

template <typename Dtype>
void AdaDeltaSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype delta = this->param_.delta();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  size_t update_history_offset = net_params.size();
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // compute square of gradient in update
    caffe_powx(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_cpu_data());

    // update history of gradients
    caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
        this->update_[param_id]->cpu_data(), momentum,
        this->history_[param_id]->mutable_cpu_data());

    // add delta to history to guard against dividing by zero later
    caffe_set(net_params[param_id]->count(), delta,
        this->temp_[param_id]->mutable_cpu_data());

    caffe_add(net_params[param_id]->count(),
        this->temp_[param_id]->cpu_data(),
        this->history_[update_history_offset + param_id]->cpu_data(),
        this->update_[param_id]->mutable_cpu_data());

    caffe_add(net_params[param_id]->count(),
        this->temp_[param_id]->cpu_data(),
        this->history_[param_id]->cpu_data(),
        this->temp_[param_id]->mutable_cpu_data());

    // divide history of updates by history of gradients
    caffe_div(net_params[param_id]->count(),
        this->update_[param_id]->cpu_data(),
        this->temp_[param_id]->cpu_data(),
        this->update_[param_id]->mutable_cpu_data());

    // jointly compute the RMS of both for update and gradient history
    caffe_powx(net_params[param_id]->count(),
        this->update_[param_id]->cpu_data(), Dtype(0.5),
        this->update_[param_id]->mutable_cpu_data());

    // compute the update
    caffe_mul(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(),
        this->update_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());

    // compute square of update
    caffe_powx(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_cpu_data());

    // update history of updates
    caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
        this->update_[param_id]->cpu_data(), momentum,
        this->history_[update_history_offset + param_id]->mutable_cpu_data());

    // apply learning rate
    caffe_cpu_scale(net_params[param_id]->count(), local_rate,
        net_params[param_id]->cpu_diff(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    // compute square of gradient in update
    caffe_gpu_powx(net_params[param_id]->count(),
        net_params[param_id]->gpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_gpu_data());

    // update history of gradients
    caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
        this->update_[param_id]->gpu_data(), momentum,
        this->history_[param_id]->mutable_gpu_data());

    // add delta to history to guard against dividing by zero later
    caffe_gpu_set(net_params[param_id]->count(), delta,
        this->temp_[param_id]->mutable_gpu_data());

    caffe_gpu_add(net_params[param_id]->count(),
        this->temp_[param_id]->gpu_data(),
        this->history_[update_history_offset + param_id]->gpu_data(),
        this->update_[param_id]->mutable_gpu_data());

    caffe_gpu_add(net_params[param_id]->count(),
        this->temp_[param_id]->gpu_data(),
        this->history_[param_id]->gpu_data(),
        this->temp_[param_id]->mutable_gpu_data());

    // divide history of updates by history of gradients
    caffe_gpu_div(net_params[param_id]->count(),
        this->update_[param_id]->gpu_data(),
        this->temp_[param_id]->gpu_data(),
        this->update_[param_id]->mutable_gpu_data());

    // jointly compute the RMS of both for update and gradient history
    caffe_gpu_powx(net_params[param_id]->count(),
        this->update_[param_id]->gpu_data(), Dtype(0.5),
        this->update_[param_id]->mutable_gpu_data());

    // compute the update and copy to net_diff
    caffe_gpu_mul(net_params[param_id]->count(),
        net_params[param_id]->gpu_diff(),
        this->update_[param_id]->gpu_data(),
        net_params[param_id]->mutable_gpu_diff());

    // compute square of update
    caffe_gpu_powx(net_params[param_id]->count(),
        net_params[param_id]->gpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_gpu_data());

    // update history of updates
    caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
        this->update_[param_id]->gpu_data(), momentum,
        this->history_[update_history_offset + param_id]->mutable_gpu_data());

    // apply learning rate
    caffe_gpu_scale(net_params[param_id]->count(), local_rate,
        net_params[param_id]->gpu_diff(),
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(AdaDeltaSolver);
REGISTER_SOLVER_CLASS(AdaDelta);

}  // namespace caffe
