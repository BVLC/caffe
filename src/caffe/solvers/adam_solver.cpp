#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void AdamSolver<Dtype>::AdamPreSolve() {
  // Add the extra history and temp entries for Adam after those from
  // SGDSolver::PreSolve
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    this->history_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    if (this->param_.amsgrad())
      this->temp_.push_back(
			    shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void adam_update_gpu(int N, int t, Dtype* g, Dtype* m, Dtype* v, const Dtype* param, Dtype beta1,
                     Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate, Dtype nu_lambda,
                     bool amsgrad, bool decoupled_wd, bool rectified);
#endif

template <typename Dtype>
void AdamSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype local_rate = rate * net_params_lr[param_id];
  const Dtype beta1 = this->param_.momentum();
  const Dtype beta2 = this->param_.momentum2();
  const bool amsgrad = this->param_.amsgrad();
  const bool rectified = this->param_.rectified();

  // we create aliases for convenience
  size_t update_history_offset = net_params.size();
  Blob<Dtype>* val_m = this->history_[param_id].get();
  Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
  Blob<Dtype>* val_t = this->temp_[param_id].get();
  Blob<Dtype>* val_v_t = this->temp_[param_id + update_history_offset].get();
  
  const int t = this->iter_ + 1;
  const Dtype correction = std::sqrt(Dtype(1) - pow(beta2, t)) /
      (Dtype(1.) - pow(beta1, t));
  const int N = net_params[param_id]->count();
  const Dtype eps_hat = this->param_.delta();

  switch (Caffe::mode()) {
    case Caffe::CPU: {
    // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
    caffe_cpu_axpby(N, Dtype(1)-beta1,
        net_params[param_id]->cpu_diff(), beta1,
        val_m->mutable_cpu_data());

    // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
    caffe_mul(N,
        net_params[param_id]->cpu_diff(),
        net_params[param_id]->cpu_diff(),
    val_t->mutable_cpu_data());

    if (amsgrad)
      for (int k=0;k<N;k++)
	{
	  val_v_t->mutable_cpu_data()[k] = val_v->cpu_data()[k];
	}
    
    caffe_cpu_axpby(N, Dtype(1)-beta2,
        val_t->cpu_data(), beta2,
        val_v->mutable_cpu_data());

    if (amsgrad)
      for (int k=0;k<N;k++)
	{
	  val_v->mutable_cpu_data()[k] = std::max(val_v_t->cpu_data()[k],
						  val_v->cpu_data()[k]);
	}
    
    // set update
    caffe_powx(N,
        val_v->cpu_data(), Dtype(0.5),
        val_t->mutable_cpu_data());
    caffe_add_scalar(N, eps_hat, val_t->mutable_cpu_data());

    if (!rectified)
      {
        caffe_div(N,
                  val_m->cpu_data(),
                  val_t->cpu_data(),
                  val_t->mutable_cpu_data());
      }
    else
      {
        Dtype rho_inf = 2.0/(1.0-beta2) - 1.0;
        Dtype rho_t = rho_inf - 2.0 * t * pow(beta2,t)/(1-pow(beta2,t)) ;
        if (rho_t > 4.0)
          {
            Dtype r_t = sqrt( (rho_t-4.0) * (rho_t-2.0) * rho_inf
                             / (rho_inf-4.0) / (rho_inf-2.0) / rho_t);
            caffe_div(N,
                      val_m->cpu_data(),
                      val_t->cpu_data(),
                      val_t->mutable_cpu_data());
            caffe_cpu_scale(N,r_t,val_t->cpu_data(),val_t->mutable_cpu_data());
          }
        else
          {
            caffe_copy(N, val_m->cpu_data(),val_t->mutable_cpu_data());
          }
      }

    if (this->param_.regularization_type() != "decoupled")
      {
        caffe_cpu_scale(N, local_rate*correction,
                        val_t->cpu_data(),
                        net_params[param_id]->mutable_cpu_diff());
      }
    else
      {
        caffe_cpu_scale(N, local_rate*correction,
                        val_t->cpu_data(),
                        net_params[param_id]->mutable_cpu_diff());
        Dtype local_decay = this->param_.weight_decay()
          * this->net_->params_weight_decay()[param_id] * local_rate / this->param_.base_lr();
        caffe_axpy(N, local_decay, net_params[param_id]->cpu_data(),
                       val_t->mutable_cpu_data());
      }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    Dtype local_decay = this->param_.weight_decay()
      * this->net_->params_weight_decay()[param_id] * local_rate / this->param_.base_lr();
    adam_update_gpu(N, t, net_params[param_id]->mutable_gpu_diff(),
                    val_m->mutable_gpu_data(), val_v->mutable_gpu_data(),
                    net_params[param_id]->gpu_data(), beta1, beta2,
                    eps_hat, local_rate * correction,  local_decay,
                    amsgrad, this->param_.regularization_type() == "decoupled", rectified);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(AdamSolver);
REGISTER_SOLVER_CLASS(Adam);

}  // namespace caffe
