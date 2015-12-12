#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void NesterovSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  CHECK(Caffe::root_solver());
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // save history momentum for stepping back
    caffe_copy(net_params[param_id]->count(),
        this->history_[param_id]->cpu_data(),
        this->update_[param_id]->mutable_cpu_data());

    // update history
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              this->history_[param_id]->mutable_cpu_data());

    // compute update: step back then over step
    caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
        this->history_[param_id]->cpu_data(), -momentum,
        this->update_[param_id]->mutable_cpu_data());

    // copy
    caffe_copy(net_params[param_id]->count(),
        this->update_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    // save history momentum for stepping back
    caffe_copy(net_params[param_id]->count(),
        this->history_[param_id]->gpu_data(),
        this->update_[param_id]->mutable_gpu_data());

    // update history
    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->gpu_diff(), momentum,
              this->history_[param_id]->mutable_gpu_data());

    // compute update: step back then over step
    caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
        this->history_[param_id]->gpu_data(), -momentum,
        this->update_[param_id]->mutable_gpu_data());

    // copy
    caffe_copy(net_params[param_id]->count(),
        this->update_[param_id]->gpu_data(),
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

INSTANTIATE_CLASS(NesterovSolver);
REGISTER_SOLVER_CLASS(Nesterov);

}  // namespace caffe
