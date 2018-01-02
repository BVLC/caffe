#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

#ifndef CPU_ONLY
template<typename Dtype>
void rmsprop_update_gpu(Device* dev, DeviceProgram* dev_prog, uint_tp n,
                        vptr<Dtype> g, vptr<Dtype> h, Dtype rms_decay,
                        Dtype delta, Dtype local_rate);
#endif

template <typename Dtype>
void RMSPropSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<BlobBase*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();

  // get the learning rate
  Dtype delta = this->param_.delta();
  Dtype rms_decay = this->param_.rms_decay();
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

      // compute square of gradient in update
      caffe_powx(net_params[param_id]->count(), cpu_diff, Dtype(2),
          this->update_[param_id]->mutable_cpu_data());

      // update history
      caffe_axpby(net_params[param_id] -> count(),
          Dtype(1-rms_decay), this->update_[param_id]->cpu_data(),
          rms_decay, this->history_[param_id]-> mutable_cpu_data());

      // prepare update
      caffe_powx(net_params[param_id]->count(),
          this->history_[param_id]->cpu_data(), Dtype(0.5),
          this->update_[param_id]->mutable_cpu_data());

      caffe_add_scalar(net_params[param_id]->count(),
          delta, this->update_[param_id]->mutable_cpu_data());

      caffe_div(net_params[param_id]->count(),
          cpu_diff, this->update_[param_id]->cpu_data(),
          this->update_[param_id]->mutable_cpu_data());

      // scale and copy
      caffe_axpby(net_params[param_id]->count(), local_rate,
          this->update_[param_id]->cpu_data(), Dtype(0), cpu_diff);

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
      rmsprop_update_gpu(this->device_, this->device_program_.get(),
          net_params[param_id]->count(), gpu_diff,
          this->history_[param_id]->mutable_gpu_data(),
          rms_decay, delta, local_rate);
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
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
  }
}

INSTANTIATE_CLASS_1T_GUARDED(RMSPropSolver, (half_fp)(float)(double));
REGISTER_SOLVER_CLASS(RMSProp);
REGISTER_SOLVER_CLASS_INST(RMSProp, (half_fp)(float)(double));

}  // namespace caffe
