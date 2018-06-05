/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"


namespace caffe {

#ifdef CAFFE_PER_LAYER_TIMINGS

#define LAYER_UPDATE_TIMING_START(index) do { \
  if (this->net()->phase() == TRAIN) { \
    this->net()->update_start_time_per_layer[index] = this->net()->timer.Duration(); \
  } \
}while(0)

#define LAYER_UPDATE_TIMING_STOP(index) do { \
  if (this->net()->phase() == TRAIN) { \
    this->net()->update_stop_time_per_layer[index] = this->net()->timer.Duration(); \
    this->net()->update_time_per_layer[index] += (this->net()->update_stop_time_per_layer[index] - this->net()->update_start_time_per_layer[index]); \
  } \
}while(0)

#else

#define LAYER_UPDATE_TIMING_START(index)
#define LAYER_UPDATE_TIMING_STOP(index)

#endif



template <typename Dtype>
Dtype SGDSolver<Dtype>::GetWarmUpLR(int cur_iter, int warmup_iter, Dtype warmup_start_lr) {
  if (cur_iter < 0) {
    cur_iter = 0;
  }
  return (cur_iter * this->param_.base_lr() +
          (warmup_iter - cur_iter) * warmup_start_lr) / warmup_iter;
}
// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();


  if (this->param_.warmup_iter() > 0 &&
      this->iter_ < this->param_.warmup_iter()) {
    rate = GetWarmUpLR(this->iter_, this->param_.warmup_iter(),
                       this->param_.warmup_start_lr());
  } else if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else if (lr_policy == "plateau") {
    // Update minimum loss if needed
    if (this->smoothed_loss_ < this->minimum_loss_) {
      this->minimum_loss_ = this->smoothed_loss_;
      this->iter_last_event_ = this->iter_;
    }

    // If sufficient iters have passed after the last event, then lower LR
    // An event is defined an update of minimum loss or LR
    if (this->current_step_ < this->param_.plateau_winsize_size()) {
      int iter_next_update = this->iter_last_event_
            + this->param_.plateau_winsize(this->current_step_);

      if (this->iter_ >= iter_next_update) {
        this->current_step_++;
        this->iter_last_event_ = this->iter_;
        LOG(INFO) << "Plateau Status: Iteration " << this->iter_
                  << ", step = " << this->current_step_;
      }
    }

    if (this->param_.display() && this->iter_ % this->param_.display() == 0
        && this->iter_last_event_ > (this->iter_ - this->param_.display())) {
      LOG(INFO) << "Plateau Status: Iteration " << this->iter_
                << ", current minimum_loss = " << this->minimum_loss_;
    }

    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "multifixed") {
      CHECK_EQ(this->param_.stageiter_size(), this->param_.stagelr_size());
      int num_stages = this->param_.stagelr_size();
      int stage = 0;
      for (; stage < num_stages; ++stage) {
          if (this->iter_ <= this->param_.stageiter(stage))
              break;
      }
      stage = (stage == num_stages) ? stage - 1 : stage;
      rate = this->param_.stagelr(stage);
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();

    // TODO: allocate these buffers taking into account owned_count to reduce memory footprint
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }

  this->minimum_loss_ = std::numeric_limits<float>::max();
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    sumsq_diff += net_params[i]->sumsq_diff();
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      net_params[i]->scale_diff(scale_factor);
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::PrintLearningRate() {
  CHECK(Caffe::root_solver());
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  PrintLearningRate();
  ClipGradients();
#ifdef CAFFE_PER_LAYER_TIMINGS
#ifdef USE_MLSL
  CHECK(mn::is_multinode() == false);
#endif
  for (int i=0; i<this->net_->layers().size(); i++) {
    const std::vector<int> param_ids = this->net_->get_layer_learnable_param_ids(i);
    LAYER_UPDATE_TIMING_START(i);
    for (int param_id = 0; param_id < param_ids.size(); ++param_id) {
      ApplyUpdate(param_ids[param_id]);
    }
    LAYER_UPDATE_TIMING_STOP(i);
  }
#else
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
    ApplyUpdate(param_id);
  }
#endif
}

template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate(int param_id) {
  CHECK(Caffe::root_solver());
  Dtype rate = GetLearningRate();

  LOG_PARAM_BLOB(this->net_->learnable_params()[param_id], diff, param_id, "ApplyUpdate: raw delwt:");

  // If Learning rate for this learnable params is zero then skip
  // updating params
  if (this->net_->params_lr()[param_id] == 0) {
    return;
  }

#ifdef ENABLE_SGD_FUSION
  if ((Caffe::mode() == Caffe::CPU) && (this->type() == string("SGD")))
  {
    //VLOG(1) << "Use Normalize_Regularize_ComputeUpdateValue_Update_Fusion for SGD";
    //LOG(INFO) << "Use Normalize_Regularize_ComputeUpdateValue_Update_Fusion for SGD";
    SGDFusion(param_id, rate);
    return;
  }
#endif /* ENABLE_SGD_FUSION */

  //LOG(INFO) << "No Fusion: Param_id: " << param_id;
  Normalize(param_id);
  
  LOG_PARAM_BLOB(this->net_->learnable_params()[param_id], diff, param_id, "ApplyUpdate: delwt after Normalize:");

  Regularize(param_id);

  LOG_PARAM_BLOB(this->net_->learnable_params()[param_id], diff, param_id, "ApplyUpdate: delwt after Regularize:");

  ComputeUpdateValue(param_id, rate);

  LOG_PARAM_BLOB(this->net_->learnable_params()[param_id], diff, param_id, "ApplyUpdate: wtinc:");

  LOG_PARAM_BLOB(this->net_->learnable_params()[param_id], data, param_id, "ApplyUpdate: weight before update:");

  this->net_->learnable_params()[param_id]->Update();

  LOG_PARAM_BLOB(this->net_->learnable_params()[param_id], data, param_id, "ApplyUpdate: weight after update:");
}

#ifdef ENABLE_SGD_FUSION
//Math function for fusion
//Function 1: axpy_axpby_copy
//Start: For L1 Regularize_ComputeUpdateValue_Fusion
template <typename Dtype>
void axpy_axpby_copy(size_t count, const Dtype decay, const Dtype* net_params_data, Dtype *net_params_diff,
                     const Dtype rate, const Dtype momentum, Dtype* history_data);

template <>
void axpy_axpby_copy<float>(size_t count, const float decay, const float* net_params_data, float *net_params_diff,
                            const float rate, const float momentum, float* history_data)
{
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static)
#endif  
  for (size_t i = 0; i < count; ++i) {
    history_data[i] = rate * (decay * net_params_data[i] + net_params_diff[i]) + momentum * history_data[i];
    net_params_diff[i] = history_data[i];
  }
}

template <>
void axpy_axpby_copy<double>(size_t count, const double decay, const double* net_params_data, double *net_params_diff,
                             const double rate, const double momentum, double* history_data)
{
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static)
#endif  
  for (size_t i = 0; i < count; ++i) {
    history_data[i] = rate * (decay * net_params_data[i] + net_params_diff[i]) + momentum * history_data[i];
    net_params_diff[i] = history_data[i];
  }
}
//End: For L1 Regularize_ComputeUpdateValue_Fusion

//Function 2: axpy_axpby_copy_axpy
//Start: For L2 Regularize_ComputeUpdateValue_Update_Fusion
template <typename Dtype>
void axpy_axpby_copy_axpy(size_t count, const Dtype decay, Dtype* net_params_data, Dtype *net_params_diff,
                     const Dtype rate, const Dtype momentum, Dtype* history_data, const Dtype update_param);

template <>
void axpy_axpby_copy_axpy<float>(size_t count, const float decay, float* net_params_data, float *net_params_diff,
                            const float rate, const float momentum, float* history_data, const float update_param)
{
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static)
#endif  
  for (size_t i = 0; i < count; ++i) {
    history_data[i] = rate * (decay * net_params_data[i] + net_params_diff[i]) + momentum * history_data[i];
    net_params_data[i] = update_param * history_data[i] + net_params_data[i];
  }
}

template <>
void axpy_axpby_copy_axpy<double>(size_t count, const double decay, double* net_params_data, double *net_params_diff,
                             const double rate, const double momentum, double* history_data, const double update_param)
{
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static)
#endif  
  for (size_t i = 0; i < count; ++i) {
    history_data[i] = rate * (decay * net_params_data[i] + net_params_diff[i]) + momentum * history_data[i];
    net_params_data[i] = update_param * history_data[i] + net_params_data[i];
  }
}
//End: For L2 Regularize_ComputeUpdateValue_Update_Fusion


template <typename Dtype>
void SGDSolver<Dtype>::SGDFusion(int param_id, Dtype rate) {
//LOG(INFO) << "Fusion: Param_id: " << param_id;

//#pragma region 1. Common initialization
  //Normalize initialization
  bool skip_Normalize_stage_flag = false;
  if (this->param_.iter_size() == 1) { skip_Normalize_stage_flag = true; }

  // Scale gradient to counterbalance accumulation.
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();

  //Regularize initialization
  const vector<float>& net_params_weight_decay =
    this->net_->params_weight_decay();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

  //ComputeUpdateValue  initialization
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * GetLocalRate(param_id);
//#pragma endregion

//#pragma region 2. Common condition judgement
  bool prv_diff_condition_flag = false;
  if (net_params[param_id]->prv_diff()
    && (net_params[param_id]->prv_diff_count()
    == net_params[param_id]->count())) {
      prv_diff_condition_flag = true;
  }
//#pragma endregion

//#pragma region 3. Normalize stage    
  if (skip_Normalize_stage_flag == false)
  {
    const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
      
    if (prv_diff_condition_flag) {
      caffe_scal(net_params[param_id]->prv_diff_count(), accum_normalization,
        net_params[param_id]->mutable_prv_diff());
    }
    else {
      caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_cpu_diff());
    }
  }
//#pragma endregion

//For most common topologies from BVLC, all skipped the Normalize stage, and use L2 regularization
//If prv_diff_condition_flag == true, then prv_data_condition_flag == true    (1)
//If prv_diff_condition_flag == false, then prv_data_condition_flag == false  (2)
//Another case is local_decay == 0, prv_diff_condition_flag == false          (3)
//So only need to consider the fusion in situations (1) and (2), set execute_separate_ComputeUpdateValue_stage_flag to false value
//We can extend the fusion in L1 regularization by axpy_axpby_copy
//We extend the fusion of Update stage in L2 regularization by axpy_axpby_copy_axpy,
//then need to change execute_separate_ComputeUpdateValue_stage_flag to execute_separate_ComputeUpdateValue_Update_stage_flag
//Simplify the execute_separate_ComputeUpdateValue_Update_stage_flag to is_separate_ComputeUpdateValue_Update
  bool is_separate_ComputeUpdateValue_Update = true;
  //Regularize stage (Fused ComputeUpdateValue_stage in some situations)
  if (local_decay) {
    if (regularization_type == "L2") {
      // add weight decay
      if (net_params[param_id]->prv_data() && net_params[param_id]->prv_diff()
        && (net_params[param_id]->prv_data_count()
        == net_params[param_id]->count())) {
          CHECK_EQ(true,
            net_params[param_id]->get_prv_data_descriptor()->layout_compare(
            net_params[param_id]->get_prv_diff_descriptor()));
          if (prv_diff_condition_flag) {
            axpy_axpby_copy_axpy(net_params[param_id]->prv_data_count(), local_decay,
                                net_params[param_id]->mutable_prv_data(), net_params[param_id]->mutable_prv_diff(),
                                local_rate, momentum, history_[param_id]->mutable_cpu_data(), Dtype(-1));

            is_separate_ComputeUpdateValue_Update = false;
          }
      } else {
        if (!prv_diff_condition_flag)
        {
          axpy_axpby_copy_axpy(net_params[param_id]->count(), local_decay,
                                net_params[param_id]->mutable_cpu_data(), net_params[param_id]->mutable_cpu_diff(),
                                local_rate, momentum, history_[param_id]->mutable_cpu_data(), Dtype(-1));

          is_separate_ComputeUpdateValue_Update = false;
        }
      }
    } else if (regularization_type == "L1") {
      caffe_cpu_sign(net_params[param_id]->count(),
                      net_params[param_id]->cpu_data(),
                      temp_[param_id]->mutable_cpu_data());

      axpy_axpby_copy(net_params[param_id]->count(), local_decay,
                                temp_[param_id]->cpu_data(), net_params[param_id]->mutable_cpu_diff(),
                                local_rate, momentum, history_[param_id]->mutable_cpu_data());
      
      is_separate_ComputeUpdateValue_Update = false;
      
      //Update stage (separate)
      net_params[param_id]->Update();
    } else {
      LOG(FATAL) << "Unknown regularization type: " << regularization_type;
    }
  }
  
  //ComputeUpdateValue_Update stage (separate)
  if (is_separate_ComputeUpdateValue_Update == true)
  {
    //Include the situation: regularization_type == "Unknown"
    //Include situations (3): local_decay == 0
    //No Regularize stage, only ComputeUpdateValue stage
    //ComputeUpdateValue stage
    if (prv_diff_condition_flag) {
      caffe_cpu_axpby(net_params[param_id]->prv_diff_count(), local_rate,
                      net_params[param_id]->prv_diff(), momentum,
                      history_[param_id]->mutable_cpu_data());

      caffe_copy(net_params[param_id]->count(),
                  history_[param_id]->cpu_data(),
                  net_params[param_id]->mutable_prv_diff());
    } else {
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                      net_params[param_id]->cpu_diff(), momentum,
                      history_[param_id]->mutable_cpu_data());

      caffe_copy(net_params[param_id]->count(),
                  history_[param_id]->cpu_data(),
                  net_params[param_id]->mutable_cpu_diff());
    }

    //Update stage (separate)
    net_params[param_id]->Update();
  }
}
#endif /* ENABLE_SGD_FUSION */

template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {

  if (this->param_.iter_size() == 1) { 
    //LOG(INFO) << "Normalize stage: Normalize stage is skipped.";
    return;
  }

  //LOG(INFO) << "Normalize stage: Normalize stage is not skipped.";
  // Scale gradient to counterbalance accumulation.
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();

  switch (Caffe::mode()) {
  case Caffe::CPU: {

    if (net_params[param_id]->prv_diff()
        && (net_params[param_id]->prv_diff_count()
            == net_params[param_id]->count())) {
        caffe_scal(net_params[param_id]->prv_diff_count(), accum_normalization,
            net_params[param_id]->mutable_prv_diff());
    }
    else {
        caffe_scal(net_params[param_id]->count(), accum_normalization,
            net_params[param_id]->mutable_cpu_diff());
    }

    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
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

template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        if (net_params[param_id]->prv_data() && net_params[param_id]->prv_diff()
             && (net_params[param_id]->prv_data_count()
                 == net_params[param_id]->count())) {
          CHECK_EQ(true,
            net_params[param_id]->get_prv_data_descriptor()->layout_compare(
            net_params[param_id]->get_prv_diff_descriptor()));

          caffe_axpy(net_params[param_id]->prv_data_count(),
                     local_decay,
                     net_params[param_id]->prv_data(),
                     net_params[param_id]->mutable_prv_diff());
        } else {
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        }
      } else if (regularization_type == "L1") {
        caffe_cpu_sign(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else if (regularization_type == "L1") {
        caffe_gpu_sign(net_params[param_id]->count(),
            net_params[param_id]->gpu_data(),
            temp_[param_id]->mutable_gpu_data());
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
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

#ifndef CPU_ONLY
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate);
#endif

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * GetLocalRate(param_id);

  if (this->param_.warmup_iter() > 0 &&
      this->iter_ < this->param_.warmup_iter()) {
    // Momentum correction during warmup stage
    Dtype prev_rate = GetWarmUpLR(this->iter_ - 1, this->param_.warmup_iter(),
                                  this->param_.warmup_start_lr());
    momentum = momentum * (rate / prev_rate);
  }
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    if (net_params[param_id]->prv_diff()
        && (net_params[param_id]->prv_diff_count()
            == net_params[param_id]->count())) {
      caffe_cpu_axpby(net_params[param_id]->prv_diff_count(), local_rate,
                      net_params[param_id]->prv_diff(), momentum,
                      history_[param_id]->mutable_cpu_data());

      caffe_copy(net_params[param_id]->count(),
                 history_[param_id]->cpu_data(),
                 net_params[param_id]->mutable_prv_diff());
    } else {
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                     net_params[param_id]->cpu_diff(), momentum,
                     history_[param_id]->mutable_cpu_data());

      caffe_copy(net_params[param_id]->count(),
                 history_[param_id]->cpu_data(),
                 net_params[param_id]->mutable_cpu_diff());

      if (net_params[param_id]->prv_diff() 
          && (net_params[param_id]->prv_diff_count()
              != net_params[param_id]->count())) {
          net_params[param_id]->mutable_prv_diff();
      }
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    sgd_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        history_[param_id]->mutable_gpu_data(),
        momentum, local_rate);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

//
// LARS (Layer-wise Adaptive Rate Scaling) is implemented by Yang You, Ignor Gitman and Boris Ginsburg in UC Berkeley.
// please refer to the papers below:
//     Scaling SGD Batch Size to 32K for ImageNet Training (https://www2.eecs.berkeley.edu/Pubs/TechRpts/2017/EECS-2017-149.html).
//     Large Batch Training of Convolutional Networks (https://arxiv.org/abs/1708.03888).
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLocalRate(int param_id) const {
  const vector<float>& net_params_lr = this->net_->params_lr();
  float local_lr = net_params_lr[param_id];

  if (this->param_.local_lr_auto()) {
    Blob<Dtype>* param = this->net_->learnable_params()[param_id];
    const float w_norm = std::sqrt(param->sumsq_data());
    const float wgrad_norm = std::sqrt(param->sumsq_diff());
    const float gw_ratio = this->param_.local_gw_ratio();
    float rate = 1.F;

    float weight_decay = this->param_.weight_decay();
    if (w_norm > 0.F && wgrad_norm > 0.F) {
      rate = gw_ratio * w_norm / (wgrad_norm + weight_decay * w_norm);
    }
    if (local_lr > 0.F) {
      local_lr = rate;
    }

#ifdef DEBUG
    if (Caffe::root_solver()
        && this->param_.display()
        && (this->iter_ % this->param_.display() == 0)) {
      const int layer_id = this->net_->param_layer_indices(param_id).first;
      const string& layer_name = this->net_->layer_names()[layer_id];
      const int blob_id = this->net_->param_layer_indices(param_id).second;
      LOG(INFO) << layer_name << "." << blob_id << " lr=" << local_lr
        << ".\t  w=" << w_norm << "\t  dw=" << wgrad_norm;
    }
#endif
  }
  return local_lr;
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
  switch (this->param_.snapshot_format()) {
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      SnapshotSolverStateToBinaryProto(model_filename);
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      SnapshotSolverStateToHDF5(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
    const string& model_filename) {
#ifdef USE_MLSL
  if (mn::is_root()) {
#endif
  SolverState state;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(this->current_step_);
  state.set_iter_last_event(this->iter_last_event_);
  state.set_minimum_loss(this->minimum_loss_);
  state.clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state.add_history();
    history_[i]->ToProto(history_blob);
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
  LOG(INFO)
    << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
#ifdef USE_MLSL
  }
#endif
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(
    const string& model_filename) {
#ifdef USE_MLSL
  if (mn::is_root()) {
#endif
  string snapshot_filename =
      Solver<Dtype>::SnapshotFilename(".solverstate.h5");
  LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
  hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC,
      H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << snapshot_filename << " to save solver state.";
  hdf5_save_int(file_hid, "iter", this->iter_);
  hdf5_save_string(file_hid, "learned_net", model_filename);
  hdf5_save_int(file_hid, "current_step", this->current_step_);
  hdf5_save_int(file_hid, "iter_last_event", this->iter_last_event_);
  hdf5_save_float<Dtype>(file_hid, "minimum_loss", this->minimum_loss_);
  hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(history_hid, 0)
      << "Error saving solver state to " << snapshot_filename << ".";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_save_nd_dataset<Dtype>(history_hid, oss.str(), *history_[i]);
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
#ifdef USE_MLSL
  }
#endif
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
    const string& state_file) {
  SolverState state;
  ReadProtoFromBinaryFile(state_file, &state);
  this->iter_ = state.iter();
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  this->iter_last_event_ = state.iter_last_event();
  this->minimum_loss_ = state.minimum_loss();
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
  hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file;
  this->iter_ = hdf5_load_int(file_hid, "iter");
  if (H5LTfind_dataset(file_hid, "learned_net")) {
    string learned_net = hdf5_load_string(file_hid, "learned_net");
    this->net_->CopyTrainedLayersFrom(learned_net);
  }
  this->current_step_ = hdf5_load_int(file_hid, "current_step");
  this->iter_last_event_ = hdf5_load_int(file_hid, "iter_last_event");
  this->minimum_loss_ = hdf5_load_float<Dtype>(file_hid, "minimum_loss");
  hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT);
  CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
  int state_history_size = hdf5_get_num_links(history_hid);
  CHECK_EQ(state_history_size, history_.size())
      << "Incorrect length of history blobs.";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_load_nd_dataset<Dtype>(history_hid, oss.str().c_str(), 0,
                                kMaxBlobAxes, history_[i].get());
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

INSTANTIATE_CLASS(SGDSolver);
REGISTER_SOLVER_CLASS(SGD);

}  // namespace caffe
