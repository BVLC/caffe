#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void RMSPropSolver<Dtype>::ComputeUpdateValue(uint_tp param_id, Dtype rate) {
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
      if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
        // compute square of gradient in update
        caffe_gpu_powx(net_params[param_id]->count(),
                       net_params[param_id]->gpu_diff(), Dtype(2),
                       this->update_[param_id]->mutable_gpu_data());

        // update history
        caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1 - rms_decay),
                        this->update_[param_id]->gpu_data(), rms_decay,
                        this->history_[param_id]->mutable_gpu_data());

        // prepare update
        caffe_gpu_powx(net_params[param_id]->count(),
                       this->history_[param_id]->gpu_data(), Dtype(0.5),
                       this->update_[param_id]->mutable_gpu_data());

        caffe_gpu_add_scalar(net_params[param_id]->count(), delta,
                             this->update_[param_id]->mutable_gpu_data());

        caffe_gpu_div(net_params[param_id]->count(),
                      net_params[param_id]->gpu_diff(),
                      this->update_[param_id]->gpu_data(),
                      this->update_[param_id]->mutable_gpu_data());

        caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
                        this->update_[param_id]->gpu_data(), Dtype(0),
                        net_params[param_id]->mutable_gpu_diff());
#endif  // USE_CUDA
      } else {
#ifdef USE_GREENTEA
        // compute square of gradient in update
        greentea_gpu_powx<Dtype>(
            this->device_->id(), net_params[param_id]->count(),
            (cl_mem) (net_params[param_id]->gpu_diff()), 0, Dtype(2),
            (cl_mem) (this->update_[param_id]->mutable_gpu_data()), 0);

        // update history
        greentea_gpu_axpby<Dtype>(
            this->device_->id(), net_params[param_id]->count(),
            Dtype(1 - rms_decay),
            (cl_mem) (this->update_[param_id]->gpu_data()), 0, rms_decay,
            (cl_mem) (this->history_[param_id]->mutable_gpu_data()), 0);

        // prepare update
        greentea_gpu_powx<Dtype>(
            this->device_->id(), net_params[param_id]->count(),
            (cl_mem) (this->history_[param_id]->gpu_data()), 0, Dtype(0.5),
            (cl_mem) (this->update_[param_id]->mutable_gpu_data()), 0);

        greentea_gpu_add_scalar<Dtype>(
            this->device_->id(), net_params[param_id]->count(), delta,
            (cl_mem) (this->update_[param_id]->mutable_gpu_data()), 0);

        greentea_gpu_div<Dtype>(
            this->device_->id(), net_params[param_id]->count(),
            (cl_mem) (net_params[param_id]->gpu_diff()), 0,
            (cl_mem) (this->update_[param_id]->gpu_data()), 0,
            (cl_mem) (this->update_[param_id]->mutable_gpu_data()), 0);

        greentea_gpu_axpby<Dtype>(
            this->device_->id(), net_params[param_id]->count(),
            local_rate, (cl_mem) (this->update_[param_id]->gpu_data()), 0,
            Dtype(0), (cl_mem) (net_params[param_id]->mutable_gpu_diff()), 0);
#endif  // USE_GREENTEA
      }
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
