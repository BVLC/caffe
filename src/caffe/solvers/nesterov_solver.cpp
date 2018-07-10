#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

#ifndef CPU_ONLY
template<typename Dtype>
void nesterov_update_gpu(Device* dev, DeviceProgram* dev_prog, uint_tp n,
                         vptr<Dtype> g, vptr<Dtype> h,
                         Dtype momentum, Dtype local_rate);
#endif

template <typename Dtype>
void NesterovSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<BlobBase*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  switch (Caffe::mode()) {
    case Caffe::CPU: {
      int_tp buffer_id = -1;
      Dtype* cpu_diff = nullptr;
      if (net_params[param_id]->data_type() == proto_data_type<Dtype>()) {
        cpu_diff = static_cast<Blob<Dtype>*>(net_params[param_id])
            ->mutable_cpu_diff();
      } else {
        cpu_diff = this->device_->template Buffer<Dtype>(
          net_params[param_id]->shape(), &buffer_id)->mutable_cpu_diff();
        net_params[param_id]->cpu_diff(cpu_diff);
      }

      // save history momentum for stepping back
      caffe_copy(net_params[param_id]->count(),
                 this->history_[param_id]->cpu_data(),
                 this->update_[param_id]->mutable_cpu_data());

      // update history
      caffe_axpby(net_params[param_id]->count(), local_rate,
                      cpu_diff, momentum,
                      this->history_[param_id]->mutable_cpu_data());

      // compute update: step back then over step
      caffe_axpby(net_params[param_id]->count(), Dtype(Dtype(1) + momentum),
                      this->history_[param_id]->cpu_data(), Dtype(-momentum),
                      this->update_[param_id]->mutable_cpu_data());

      // copy
      caffe_copy(net_params[param_id]->count(),
                 this->update_[param_id]->cpu_data(),
                 cpu_diff);

      if (net_params[param_id]->data_type() != proto_data_type<Dtype>()) {
        net_params[param_id]->set_cpu_diff(cpu_diff);
        this->device_->unlock_buffer(&buffer_id);
      }
      break;
    }
    case Caffe::GPU: {
#ifndef CPU_ONLY
      int_tp buffer_id = -1;
      vptr<Dtype> gpu_diff;
      if (net_params[param_id]->data_type() == proto_data_type<Dtype>()) {
        gpu_diff = static_cast<Blob<Dtype>*>(net_params[param_id])
            ->mutable_gpu_diff();
      } else {
        gpu_diff = this->device_->template Buffer<Dtype>(
          net_params[param_id]->shape(), &buffer_id)->mutable_gpu_diff();
        net_params[param_id]->gpu_diff(gpu_diff);
      }
      nesterov_update_gpu(this->device_, this->device_program_.get(),
          net_params[param_id]->count(), gpu_diff,
          this->history_[param_id]->mutable_gpu_data(),
          momentum, local_rate);
      if (net_params[param_id]->data_type() != proto_data_type<Dtype>()) {
        net_params[param_id]->set_gpu_diff(gpu_diff);
        this->device_->unlock_buffer(&buffer_id);
      }
#else
      NO_GPU;
#endif
      break;
    }
    default: {
      LOG(FATAL)<< "Unknown caffe mode: " << Caffe::mode();
    }
  }
}

INSTANTIATE_CLASS_1T_GUARDED(NesterovSolver, (half_fp)(float)(double));
REGISTER_SOLVER_CLASS(Nesterov);
REGISTER_SOLVER_CLASS_INST(Nesterov, (half_fp)(float)(double));

}  // namespace caffe
