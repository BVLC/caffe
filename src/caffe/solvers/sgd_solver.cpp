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
#include <immintrin.h>


namespace caffe {
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
void SGDSolver<Dtype>::ApplyUpdate() {
  CHECK(Caffe::root_solver());
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  ClipGradients();
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
    ApplyUpdate(param_id);
  }
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
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    //VLOG(1) << "Use Normalize_Regularize_ComputeUpdateValue_Fusion for SGD";
    //LOG(INFO) << "Use Normalize_Regularize_ComputeUpdateValue_Fusion for SGD";
    Normalize_Regularize_ComputeUpdateValue_Fusion(param_id, rate);
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    //VLOG(1) << "Currently we do not support use Normalize_Regularize_ComputeUpdateValue_Fusion for SGD in GPU mode.";
    //LOG(INFO) << "Currently we do not support use Normalize_Regularize_ComputeUpdateValue_Fusion for SGD in GPU mode.";
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
#else /* !ENABLE_SGD_FUSION */
  //LOG(INFO) << "No Fusion: Param_id: " << param_id;
  Normalize(param_id);
  
  LOG_PARAM_BLOB(this->net_->learnable_params()[param_id], diff, param_id, "ApplyUpdate: delwt after Normalize:");

  Regularize(param_id);
  LOG_PARAM_BLOB(this->net_->learnable_params()[param_id], diff, param_id, "ApplyUpdate: delwt after Regularize:");

  ComputeUpdateValue(param_id, rate);
#endif /* ENABLE_SGD_FUSION */

  LOG_PARAM_BLOB(this->net_->learnable_params()[param_id], diff, param_id, "ApplyUpdate: wtinc:");

  LOG_PARAM_BLOB(this->net_->learnable_params()[param_id], data, param_id, "ApplyUpdate: weight before update:");

  this->net_->learnable_params()[param_id]->Update();

  LOG_PARAM_BLOB(this->net_->learnable_params()[param_id], data, param_id, "ApplyUpdate: weight after update:");
}

#ifdef ENABLE_SGD_FUSION
//Math function for fusion
template <typename Dtype>
void axpy_axpby_copy(size_t count, const Dtype decay, const Dtype* net_params_data, Dtype *net_params_diff,
                     const Dtype rate, const Dtype momentum, Dtype* history_data);

template <>
void axpy_axpby_copy<float>(size_t count, const float decay, const float* net_params_data, float *net_params_diff,
                            const float rate, const float momentum, float* history_data)
{
  float temp_result = 0.;
#ifdef _OPENMP
#pragma omp parallel for
#endif  
  for (size_t i = 0; i < count; ++i) {
    temp_result = rate * (decay * net_params_data[i] + net_params_diff[i]) + momentum * history_data[i];
    history_data[i] =  temp_result;
    net_params_diff[i] =  temp_result;
  }
}

template <>
void axpy_axpby_copy<double>(size_t count, const double decay, const double* net_params_data, double *net_params_diff,
                             const double rate, const double momentum, double* history_data)
{
  double temp_result = 0.;
#ifdef _OPENMP
#pragma omp parallel for
#endif  
  for (size_t i = 0; i < count; ++i) {
    temp_result = rate * (decay * net_params_data[i] + net_params_diff[i]) + momentum * history_data[i];
    history_data[i] =  temp_result;
    net_params_diff[i] =  temp_result;
  }
}

template <typename Dtype>
void avx512_axpy_axpby_copy(size_t count, const Dtype decay, const Dtype* net_params_data, Dtype *net_params_diff,
                            const Dtype rate, const Dtype momentum, Dtype* history_data);

template <>
void avx512_axpy_axpby_copy<float>(size_t count, const float decay, const float* net_params_data, float *net_params_diff,
                                  const float rate, const float momentum, float* history_data)
{
    // If count is smaller than 16 we use non-avx512 implementation
    // 16 is the element number which one avx512 register can hold
    if (count < 16) {
        return axpy_axpby_copy(count, decay, net_params_data, net_params_diff,
                                     rate, momentum, history_data);
    }

    // If count can't be divided by 16, we handle tailing remainder
    // with non-avx512 imeplementation
    if (count % 16 != 0) {
        size_t remainder = count % 16;
        count -= remainder;
        axpy_axpby_copy(remainder, decay, net_params_data+count, net_params_diff+count,
                              rate, momentum, history_data+count);
    }

    size_t group_size = 16;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t idx = 0; idx < count; idx += group_size) {
        const float *fnet_params_data  = net_params_data + idx;
        float *fnet_params_diff        = net_params_diff + idx;
        float *fhistory_data           = history_data    + idx;
        __m512 operand1_v              = _mm512_loadu_ps(fnet_params_data);
        __m512 operand2_v              = _mm512_loadu_ps(fnet_params_diff);
        __m512 operand3_v              = _mm512_loadu_ps(fhistory_data);
        __m512 decay_operand_v         = _mm512_set1_ps(decay);
        __m512 rate_operand_v          = _mm512_set1_ps(rate);
        __m512 momentum_operand_v      = _mm512_set1_ps(momentum);
        __m512 decay_result            = _mm512_mul_ps(decay_operand_v, operand1_v);
        __m512 axpy_result             = _mm512_add_ps(decay_result, operand2_v);
        __m512 rate_result             = _mm512_mul_ps(rate_operand_v, axpy_result);
        __m512 momentum_result         = _mm512_mul_ps(momentum_operand_v, operand3_v);
        __m512 axpby_result            = _mm512_add_ps(rate_result, momentum_result);
        _mm512_storeu_ps(fhistory_data, axpby_result);
        _mm512_storeu_ps(fnet_params_diff, axpby_result);
    }
}

template <>
void avx512_axpy_axpby_copy<double>(size_t count, const double decay, const double* net_params_data, double* net_params_diff,
                                    const double rate, const double momentum, double* history_data)
{
    // If count is smaller than 8 we use non-avx512 implementation
    // 8 is the element number which one avx512 register can hold
    if (count < 8) {
        return axpy_axpby_copy(count, decay, net_params_data, net_params_diff,
                               rate, momentum, history_data);
    }

    // If count can't be divided by 8, we handle tailing remainder
    // with non-avx512 imeplementation
    if (count % 8 != 0) {
        size_t remainder = count % 8;
        count -= remainder;
        axpy_axpby_copy(remainder, decay, net_params_data+count, net_params_diff+count,
                        rate, momentum, history_data+count);
    }

    size_t group_size = 8;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t idx = 0; idx < count; idx += group_size) {
        const double *fnet_params_data  = net_params_data + idx;
        double *fnet_params_diff        = net_params_diff + idx;
        double *fhistory_data           = history_data    + idx;
        __m512 operand1_v               = _mm512_loadu_pd(fnet_params_data);
        __m512 operand2_v               = _mm512_loadu_pd(fnet_params_diff);
        __m512 operand3_v               = _mm512_loadu_pd(fhistory_data);
        __m512 decay_operand_v          = _mm512_set1_pd(decay);
        __m512 rate_operand_v           = _mm512_set1_pd(rate);
        __m512 momentum_operand_v       = _mm512_set1_pd(momentum);
        __m512 decay_result             = _mm512_mul_pd(decay_operand_v, operand1_v);
        __m512 axpy_result              = _mm512_add_pd(decay_result, operand2_v);
        __m512 rate_result              = _mm512_mul_pd(rate_operand_v, axpy_result);
        __m512 momentum_result          = _mm512_mul_pd(momentum_operand_v, operand3_v);
        __m512 axpby_result             = _mm512_add_pd(rate_result, momentum_result);
        _mm512_storeu_pd(fhistory_data, axpby_result);
        _mm512_storeu_pd(fnet_params_diff, axpby_result);
    }
}


template <typename Dtype>
void SGDSolver<Dtype>::Normalize_Regularize_ComputeUpdateValue_Fusion(int param_id, Dtype rate) {
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
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
//#pragma endregion

//#pragma region 2. Common condition judgement
  bool prv_diff_condition_flag = false;
  if (net_params[param_id]->prv_diff()
    && (net_params[param_id]->prv_diff_count()
    == net_params[param_id]->count())) {
      prv_diff_condition_flag = true;
      //LOG(INFO) << "Common condition judgement: prv_diff_condition_flag = true.";
  }
  else
  {
    //LOG(INFO) << "Common condition judgement: prv_diff_condition_flag = false.";
  }
//#pragma endregion

//#pragma region 3. Normalize stage    
  if (skip_Normalize_stage_flag == false)
  {
    //LOG(INFO) << "Normalize stage: Normalize stage is not skipped.";

    const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
      
    if (prv_diff_condition_flag) {
      //LOG(INFO) << "Normalize stage: prv_diff_condition_flag = true.";
      caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_prv_diff());
    }
    else {
      //LOG(INFO) << "Normalize stage: prv_diff_condition_flag = false.";
      caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_cpu_diff());
    }
  }
  else
  {
    //LOG(INFO) << "Normalize stage: Normalize stage is skipped.";
  }
//#pragma endregion

//For POR topologies from BVLC, all skipped the Normalize stage, and use L2 regularization
//If prv_diff_condition_flag == true, then prv_data_condition_flag == true    (1)
//If prv_diff_condition_flag == false, then prv_data_condition_flag == false  (2)
//Another case is local_decay == 0, prv_diff_condition_flag == false          (3)
//So only need to consider the fusion in situations (1) and (2), set execute_separate_ComputeUpdateValue_stage_flag to false value
  bool execute_separate_ComputeUpdateValue_stage_flag = true;
  //Regularize stage (Fused ComputeUpdateValue_stage in some situations)
  if (local_decay) {
    if (regularization_type == "L2") {
      //LOG(INFO) << "Regularize stage: regularization_type == L2.";
      // add weight decay
      if (net_params[param_id]->prv_data()
        && (net_params[param_id]->prv_data_count()
        == net_params[param_id]->count())) {
        //LOG(INFO) << "Regularize stage: prv_data_condition_flag = true.";
          CHECK_EQ(true,
            net_params[param_id]->get_prv_data_descriptor()->layout_compare(
            net_params[param_id]->get_prv_diff_descriptor()));
          /*  
          caffe_axpy(net_params[param_id]->count(), 
                      local_decay,
                      net_params[param_id]->prv_data(),
                      net_params[param_id]->mutable_prv_diff());
          */
          if (prv_diff_condition_flag) {
            //situation (1)
            //LOG(INFO) << "Fused ComputeUpdateValue stage: prv_diff_condition_flag = true.";
            /*
            caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                            net_params[param_id]->prv_diff(), momentum,
                            history_[param_id]->mutable_cpu_data());

            caffe_copy(net_params[param_id]->count(),
                        history_[param_id]->cpu_data(),
                        net_params[param_id]->mutable_prv_diff());
            */

            avx512_axpy_axpby_copy(net_params[param_id]->count(), local_decay,
                                net_params[param_id]->prv_data(), net_params[param_id]->mutable_prv_diff(),
                                local_rate, momentum, history_[param_id]->mutable_cpu_data());

            execute_separate_ComputeUpdateValue_stage_flag = false;
          }
          else
          {
            //Will not happen!
            //LOG(INFO) << "Cannot Fused ComputeUpdateValue stage: prv_diff_condition_flag = false.";
            caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                      net_params[param_id]->cpu_diff(), momentum,
                      history_[param_id]->mutable_cpu_data());

            caffe_copy(net_params[param_id]->count(),
                        history_[param_id]->cpu_data(),
                        net_params[param_id]->mutable_cpu_diff());

            execute_separate_ComputeUpdateValue_stage_flag = false;
            //You can set the flag to true value, and not execute caffe_cpu_axpby and caffe_copy
            //But set to false value and execute caffe_cpu_axpby and caffe_copy inside will save one condition judgement time
          }
      } else {
        //LOG(INFO) << "Regularize stage: prv_data_condition_flag = false.";
        /*
        caffe_axpy(net_params[param_id]->count(),
                    local_decay,
                    net_params[param_id]->cpu_data(),
                    net_params[param_id]->mutable_cpu_diff());
        */
        if (!prv_diff_condition_flag)
        {
          //situation (2)
          //LOG(INFO) << "Fused ComputeUpdateValue stage: prv_diff_condition_flag = false.";
          /*
          caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                    net_params[param_id]->cpu_diff(), momentum,
                    history_[param_id]->mutable_cpu_data());

          caffe_copy(net_params[param_id]->count(),
                      history_[param_id]->cpu_data(),
                      net_params[param_id]->mutable_cpu_diff());
          */

          avx512_axpy_axpby_copy(net_params[param_id]->count(), local_decay,
                                net_params[param_id]->cpu_data(), net_params[param_id]->mutable_cpu_diff(),
                                local_rate, momentum, history_[param_id]->mutable_cpu_data());

          execute_separate_ComputeUpdateValue_stage_flag = false;
        }
        else
        {
          //Will not happen!
          //LOG(INFO) << "Cannot Fused ComputeUpdateValue stage: prv_diff_condition_flag = true.";
          caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                          net_params[param_id]->prv_diff(), momentum,
                          history_[param_id]->mutable_cpu_data());

          caffe_copy(net_params[param_id]->count(),
                      history_[param_id]->cpu_data(),
                      net_params[param_id]->mutable_prv_diff());

          execute_separate_ComputeUpdateValue_stage_flag = false;
          //You can set the flag to true value, and not execute caffe_cpu_axpby and caffe_copy
          //But set to false value and execute caffe_cpu_axpby and caffe_copy inside will save one condition judgement time
        }        
      }
    } else if (regularization_type == "L1") {
      //LOG(INFO) << "Regularize stage: regularization_type == L1.";
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
  
  //ComputeUpdateValue stage (separate)
  if (execute_separate_ComputeUpdateValue_stage_flag == true)
  {
    //Include the situation: regularization_type == "L1"/"Unknown"
    //Include situations (3): local_decay == 0
    //No Regularize stage, only ComputeUpdateValue stage
    //ComputeUpdateValue stage
    if (prv_diff_condition_flag) {
      //LOG(INFO) << "ComputeUpdateValue stage: prv_diff_condition_flag = true.";
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                      net_params[param_id]->prv_diff(), momentum,
                      history_[param_id]->mutable_cpu_data());

      caffe_copy(net_params[param_id]->count(),
                  history_[param_id]->cpu_data(),
                  net_params[param_id]->mutable_prv_diff());
    } else {
      //LOG(INFO) << "ComputeUpdateValue stage: prv_diff_condition_flag = false.";
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                      net_params[param_id]->cpu_diff(), momentum,
                      history_[param_id]->mutable_cpu_data());

      caffe_copy(net_params[param_id]->count(),
                  history_[param_id]->cpu_data(),
                  net_params[param_id]->mutable_cpu_diff());
    }
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
        //LOG(INFO) << "Normalize stage: prv_diff_condition_flag = true.";
        caffe_scal(net_params[param_id]->count(), accum_normalization,
            net_params[param_id]->mutable_prv_diff());
    }
    else {
        //LOG(INFO) << "Normalize stage: prv_diff_condition_flag = false.";
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
        //LOG(INFO) << "Regularize stage: regularization_type == L2.";
        // add weight decay
        if (net_params[param_id]->prv_data()
             && (net_params[param_id]->prv_data_count()
                 == net_params[param_id]->count())) {
          //LOG(INFO) << "Regularize stage: prv_data_condition_flag = true.";
          CHECK_EQ(true,
            net_params[param_id]->get_prv_data_descriptor()->layout_compare(
            net_params[param_id]->get_prv_diff_descriptor()));

          caffe_axpy(net_params[param_id]->count(),
                     local_decay,
                     net_params[param_id]->prv_data(),
                     net_params[param_id]->mutable_prv_diff());
        } else {
          //LOG(INFO) << "Regularize stage: prv_data_condition_flag = false.";
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        }
      } else if (regularization_type == "L1") {
        //LOG(INFO) << "Regularize stage: regularization_type == L1.";
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
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];

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
      //LOG(INFO) << "ComputeUpdateValue stage: prv_diff_condition_flag = true.";
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                      net_params[param_id]->prv_diff(), momentum,
                      history_[param_id]->mutable_cpu_data());

      caffe_copy(net_params[param_id]->count(),
                 history_[param_id]->cpu_data(),
                 net_params[param_id]->mutable_prv_diff());
    } else {
      //LOG(INFO) << "ComputeUpdateValue stage: prv_diff_condition_flag = false.";
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                     net_params[param_id]->cpu_diff(), momentum,
                     history_[param_id]->mutable_cpu_data());

      caffe_copy(net_params[param_id]->count(),
                 history_[param_id]->cpu_data(),
                 net_params[param_id]->mutable_cpu_diff());
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
#ifdef USE_MLSL
  if (mn::is_root()) {
#endif
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
