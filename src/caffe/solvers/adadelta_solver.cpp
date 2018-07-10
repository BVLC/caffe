#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void AdaDeltaSolver<Dtype>::AdaDeltaPreSolve() {
  // Add the extra history entries for AdaDelta after those from
  // SGDSolver::PreSolve
  const vector<BlobBase*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
        const vector<int_tp>& shape = net_params[i]->shape();
        this->history_.push_back(
         shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape, this->device_)));
  }
}

#ifndef CPU_ONLY
template<typename Dtype>
void adadelta_update_gpu(Device* dev, DeviceProgram* dev_prog, uint_tp n,
                         vptr<Dtype> g, vptr<Dtype> h, vptr<Dtype> h2,
                         Dtype momentum, Dtype delta, Dtype local_rate);
#endif

template <typename Dtype>
void AdaDeltaSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<BlobBase*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype delta = this->param_.delta();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  size_t update_history_offset = net_params.size();
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
      caffe_powx(net_params[param_id]->count(),
          cpu_diff, Dtype(2),
          this->update_[param_id]->mutable_cpu_data());

      // update history of gradients
      caffe_axpby(net_params[param_id]->count(), Dtype(Dtype(1) - momentum),
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
      caffe_mul(net_params[param_id]->count(), cpu_diff,
          this->update_[param_id]->cpu_data(), cpu_diff);

      // compute square of update
      caffe_powx(net_params[param_id]->count(),
          cpu_diff, Dtype(2),
          this->update_[param_id]->mutable_cpu_data());

      // update history of updates
      caffe_axpby(net_params[param_id]->count(), Dtype(Dtype(1) - momentum),
          this->update_[param_id]->cpu_data(), momentum,
          this->history_[update_history_offset + param_id]->mutable_cpu_data());

      // apply learning rate
      caffe_scale(net_params[param_id]->count(), local_rate, cpu_diff,
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
      adadelta_update_gpu(this->device_, this->device_program_.get(),
          net_params[param_id]->count(), gpu_diff,
          this->history_[param_id]->mutable_gpu_data(),
          this->history_[update_history_offset + param_id]->mutable_gpu_data(),
          momentum, delta, local_rate);
      if (net_params[param_id]->data_type() != proto_data_type<Dtype>()) {
        net_params[param_id]->set_gpu_diff(gpu_diff);
        this->device_->unlock_buffer(&buffer_id);
      }
#else
      NO_GPU;
#endif
      break;
    }
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS_1T_GUARDED(AdaDeltaSolver, (half_fp)(float)(double));
REGISTER_SOLVER_CLASS(AdaDelta);
REGISTER_SOLVER_CLASS_INST(AdaDelta, (half_fp)(float)(double));

}  // namespace caffe
