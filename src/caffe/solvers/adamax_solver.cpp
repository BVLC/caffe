#include <algorithm>
#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void AdaMaxSolver<Dtype>::AdaMaxPreSolve() {
  // Essentially the same as with Adam
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    this->history_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void adamax_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, Dtype beta1,
    Dtype beta2, Dtype corrected_local_rate);
#endif

template <typename Dtype>
void AdaMaxSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype local_rate = rate * net_params_lr[param_id];
  const Dtype beta1 = this->param_.momentum();
  const Dtype beta2 = this->param_.momentum2();

  // we create aliases for convenience
  size_t update_history_offset = net_params.size();
  Blob<Dtype>* val_m = this->history_[param_id].get();
  Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
  Blob<Dtype>* val_t = this->temp_[param_id].get();

  const int t = this->iter_ + 1;
  const Dtype correction = Dtype(1) / (Dtype(1) - pow(beta1, t));
  const int N = net_params[param_id]->count();

  switch (Caffe::mode()) {
    case Caffe::CPU: {
    // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
    caffe_cpu_axpby(N, Dtype(1)-beta1,
        net_params[param_id]->cpu_diff(), beta1,
        val_m->mutable_cpu_data());

    // update v <- max(\beta_2 v_{t-1}, |g_t|)
    // for stability, add a small epsilon to \beta_2 v_{t-1}
    caffe_abs(N, net_params[param_id]->cpu_diff(), val_t->mutable_cpu_data());
    for (int i = 0; i < N; ++i) {
      val_v->mutable_cpu_data()[i] = std::max(
          val_v->cpu_data()[i] * beta2 + Dtype(1e-7),
          val_t->cpu_data()[i]);
    }

    // set update
    caffe_div(N,
        val_m->cpu_data(),
        val_v->cpu_data(),
        val_t->mutable_cpu_data());
    caffe_cpu_scale(N, local_rate*correction, val_t->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());

    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    adamax_update_gpu(N, net_params[param_id]->mutable_gpu_diff(),
        val_m->mutable_gpu_data(), val_v->mutable_gpu_data(), beta1, beta2,
        local_rate*correction);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(AdaMaxSolver);
REGISTER_SOLVER_CLASS(AdaMax);

}  // namespace caffe
