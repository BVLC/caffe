#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void AdamSolver<Dtype>::AdamPreSolve() {
  // Add the extra history entries for Adam after those from
  // SGDSolver::PreSolve
  const vector<BlobBase*>& net_params = this->net_->learnable_params();
  for (uint_tp i = 0; i < net_params.size(); ++i) {
    const vector<int_tp>& shape = net_params[i]->shape();
    this->history_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape, this->device_)));
  }
}

#ifndef CPU_ONLY
template<typename Dtype>
void adam_update_gpu(Device* dev, DeviceProgram* dev_prog, uint_tp n,
                     vptr<Dtype> g, vptr<Dtype> m, vptr<Dtype> v,
                     Dtype beta1, Dtype beta2, Dtype eps_hat,
                     Dtype corrected_local_rate);
#endif

template <typename Dtype>
void AdamSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<BlobBase*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype local_rate = rate * net_params_lr[param_id];
  const Dtype beta1 = this->param_.momentum();
  const Dtype beta2 = this->param_.momentum2();

  // we create aliases for convenience
  uint_tp update_history_offset = net_params.size();
  Blob<Dtype>* val_m = this->history_[param_id].get();
  Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
  Blob<Dtype>* val_t = this->temp_[param_id].get();

  const uint_tp t = this->iter_  + 1;
  const Dtype correction = std::sqrt(Dtype(1) - pow(beta2, Dtype(t))) /
      (Dtype(1.) - pow(beta1, Dtype(t)));
  const uint_tp n = net_params[param_id]->count();
  const Dtype eps_hat = this->param_.delta();

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

      // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
      caffe_axpby(n, Dtype(Dtype(1)-beta1), cpu_diff, beta1,
          val_m->mutable_cpu_data());

      // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
      caffe_mul(n, cpu_diff, cpu_diff, val_t->mutable_cpu_data());
      caffe_axpby(n, Dtype(Dtype(1)-beta2),
          val_t->cpu_data(), beta2,
          val_v->mutable_cpu_data());

      // set update
      caffe_powx(n, val_v->cpu_data(), Dtype(0.5), val_t->mutable_cpu_data());
      caffe_add_scalar(n, eps_hat, val_t->mutable_cpu_data());
      caffe_div(n, val_m->cpu_data(), val_t->cpu_data(),
                val_t->mutable_cpu_data());

      caffe_scale(n, Dtype(local_rate*correction), val_t->cpu_data(),
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
      adam_update_gpu(this->device_, this->device_program_.get(), n,
                      gpu_diff, val_m->mutable_gpu_data(),
                      val_v->mutable_gpu_data(), beta1, beta2, eps_hat,
                      Dtype(local_rate * correction));
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

INSTANTIATE_CLASS_1T_GUARDED(AdamSolver, (half_fp)(float)(double));
REGISTER_SOLVER_CLASS(Adam);
REGISTER_SOLVER_CLASS_INST(Adam, (half_fp)(float)(double));

}  // namespace caffe
