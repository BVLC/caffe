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

#ifdef USE_MLSL

#include <vector>

#include <boost/make_shared.hpp>
#include <boost/bind.hpp>

#include "caffe/multinode/multi_solver.hpp"

namespace caffe {

#ifdef CAFFE_PER_LAYER_TIMINGS

#define LAYER_TIMING_START() do { \
  root_solver_->timer.Start(); \
}while(0)

#define LAYER_TIMING_STOP(name, index) do { \
  root_solver_->name##_time_per_layer[index] += root_solver_->timer.MicroSeconds(); \
}while(0)

#ifdef FW_OVERLAP_OPT
#define LAYER_WAIT_TIMING_START() do { \
  root_solver_->wait_timer.Start(); \
}while(0)

#define LAYER_WAIT_TIMING_STOP(layer_index) do { \
  root_solver_->waitcomm_time_per_layer[layer_index] += root_solver_->wait_timer.MicroSeconds(); \
}while(0)

#define LAYER_REMOVE_UPDATE_TIME(layer_i, layer_k) do { \
  root_solver_->waitcomm_time_per_layer[layer_i] -= root_solver_->update_time_per_layer[layer_k]; \
} while (0)
#endif

#define ITER_TIMING_START() do { \
  root_solver_->timer.Start(); \
}while(0)

#define ITER_TIMING_STOP(name) do { \
  root_solver_->name##_time_per_iter += root_solver_->timer.MicroSeconds(); \
}while(0)

#else

#define LAYER_TIMING_START()
#define LAYER_TIMING_STOP(name,index)

#ifdef FW_OVERLAP_OPT
#define LAYER_WAIT_TIMING_START()
#define LAYER_WAIT_TIMING_STOP(index)
#define LAYER_REMOVE_UPDATE_TIME(layer_i, layer_k)
#endif

#define ITER_TIMING_START()
#define ITER_TIMING_STOP(name, index)

#endif

template <typename Dtype>
inline bool MultiSolver<Dtype>::IsSkipWaitGradient(int layer_id) {
  Net<Dtype>& net = *root_solver_->net();
  const std::vector<shared_ptr<Layer<Dtype>>>& layers{ net.layers() };
  const std::vector<bool>& layer_need_backward{ net.layer_need_backward() };

  if (!layer_need_backward[layer_id] || ((layers[layer_id]->layerOp != nullptr)
        && !layers[layer_id]->layerOp->HasParameterSets())) {
    DLOG(INFO) << "ForwardBackwardImpl: no need for apply_updates for layer # "
      << layer_id << ", skip on_delwt_wait, apply_updates, on_wtinc_ready";
    return true;
  }
  return false;
}

template <typename Dtype>
inline bool MultiSolver<Dtype>::WaitGradient(int layer_id) {
#ifndef FW_OVERLAP_OPT
  LAYER_TIMING_START();
#endif
  for (int j = 0; j < callbacks_.size(); ++j) {
    callbacks_[j]->on_delwt_wait(layer_id);
  }
#ifndef FW_OVERLAP_OPT
  LAYER_TIMING_STOP(waitcomm, layer_id);
  return true;
#else
  return layer_finished_flags_[layer_id];
#endif
}

template <typename Dtype>
inline void MultiSolver<Dtype>::UpdateGradient(int layer_id) {
#ifdef FW_OVERLAP_OPT
  if (layer_finished_flags_[layer_id]) {
#endif
    LAYER_TIMING_START();
    for (int j = 0; j < callbacks_.size(); ++j) {
      callbacks_[j]->apply_updates(layer_id);
    }
    LAYER_TIMING_STOP(update, layer_id);
#ifdef FW_OVERLAP_OPT
  }
#endif
}

template <typename Dtype>
Dtype MultiSolver<Dtype>::ForwardBackwardImpl(bool first, bool last) {
  Dtype loss = 0;
  Net<Dtype>& net = *root_solver_->net();
  const std::vector<shared_ptr<Layer<Dtype>>>& layers{ net.layers() };
  const std::vector<bool>& layer_need_backward{ net.layer_need_backward() };

  for (int i = 0; i < layers.size(); ++i) {
#ifdef FW_OVERLAP_OPT
    if (first) {
      LAYER_WAIT_TIMING_START();
      while (layer_finished_flags_[i] == false) {
        if (IsSkipWaitGradient(i) || WaitGradient(i))
          break;
        
        // wait and update gradient for next layers
        for (int k=i+1; k<layers.size(); k++) {
          if (layer_finished_flags_[k] || IsSkipWaitGradient(k)) {
            layer_finished_flags_[k] = true;
            continue;
          }
          if (WaitGradient(k)) {
            UpdateGradient(k);
            // The update time for layer k must be removed from waitcomm time
            // for layer i
            LAYER_REMOVE_UPDATE_TIME(i, k);
            break;
          }
        }
      }
      LAYER_WAIT_TIMING_STOP(i);
      UpdateGradient(i);
      // set flag to false after updating gradient
      layer_finished_flags_[i] = false;
    }
#endif

    LAYER_TIMING_START();
    loss += net.ForwardFromTo(i, i);
    LAYER_TIMING_STOP(forward, i);
  }

  // Clear parameter diffs after communication is finished (that is, after 
  // calling WaitGradientComm)
  if (first) {
    ITER_TIMING_START();
    root_solver_->net()->ClearParamDiffs();
    ITER_TIMING_STOP(cleardiffs);
  }

  for (int i = layers.size() - 1; i >= 0; --i) {
    if (!layer_need_backward[i]) {
      continue;
    }

    LAYER_TIMING_START();
    net.BackwardFromTo(i, i);
    LAYER_TIMING_STOP(backward, i);

    if (last && (layers[i]->layerOp != nullptr)
        && layers[i]->layerOp->HasParameterSets()) {
      LAYER_TIMING_START();
      for (int j = 0; j < callbacks_.size(); ++j) {
        callbacks_[j]->on_backward_finished(i);
      }
      LAYER_TIMING_STOP(startcomm, i);
    }
  }


#ifdef FW_OVERLAP_OPT
  int iter = root_solver_->iter();
  int max_iter = root_solver_->param().max_iter();
  bool test = (root_solver_->param().test_interval()
      && ((iter + 1) % root_solver_->param().test_interval() == 0));
  bool last_iter_wait_flag = last && (test || (iter == max_iter - 1));
#else
  bool last_iter_wait_flag = last;
#endif

  if (last_iter_wait_flag) {
    for (int i = 0; i < layers.size(); ++i) {
      if (IsSkipWaitGradient(i))
        continue;

#ifdef FW_OVERLAP_OPT
      LAYER_WAIT_TIMING_START();
      while (
#endif
        WaitGradient(i)
#ifdef FW_OVERLAP_OPT
          == false)
#endif
      ;

#ifdef FW_OVERLAP_OPT
      LAYER_WAIT_TIMING_STOP(i);
#endif
      UpdateGradient(i);
    }
  }

  DLOG(WARNING) << "iter " << root_solver_->iter() << ", loss " << loss;
  return loss;
}

template <typename Dtype>
Dtype MultiSolver<Dtype>::ForwardBackward() {
  Dtype loss = 0;
  for (int i = 0; i < iter_size; ++i) {
    loss += ForwardBackwardImpl(
        (i == 0), (i + 1 == iter_size));
  }
  return loss / iter_size;
}

INSTANTIATE_CLASS(MultiSolver);

}  // namespace caffe

#endif /* USE_MLSL */
