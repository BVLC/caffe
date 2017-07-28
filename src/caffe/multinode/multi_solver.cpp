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
#else
#define LAYER_TIMING_START()

#define LAYER_TIMING_STOP(name,index)
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
inline void MultiSolver<Dtype>::WaitAndUpdateGradient(int layer_id) {
  LAYER_TIMING_START();
  for (int j = 0; j < callbacks_.size(); ++j) {
    callbacks_[j]->on_delwt_wait(layer_id);
  }
  LAYER_TIMING_STOP(waitcomm, layer_id);

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
    if (first && IsSkipWaitGradient(i) == false) {
      while (layer_finished_flags_[i] == false) {
        WaitAndUpdateGradient(i);
        if (layer_finished_flags_[i])
          break;

        for (int k=i+1; k<layers.size(); k++) {
          if (layer_finished_flags_[k] || IsSkipWaitGradient(k)) {
            layer_finished_flags_[k] = true;
            continue;
          }
          WaitAndUpdateGradient(k);
          if (layer_finished_flags_[k])
            break;
        }
      }
    }
#endif

    LAYER_TIMING_START();
    loss += net.ForwardFromTo(i, i);
    LAYER_TIMING_STOP(forward, i);
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
        callbacks_[j]->on_iter_finished(i);
      }
      LAYER_TIMING_STOP(startcomm, i);
    }
  }

#ifdef FW_OVERLAP_OPT
  int iter = root_solver_->iter();
  int max_iter = root_solver_->param().max_iter();
  bool test = (root_solver_->param().test_interval()
          && ((iter + 1) % root_solver_->param().test_interval() == 0));
  if (last && (test || (iter == max_iter - 1))) {
    int finished_count = 0;
    while (finished_count < layers.size()) {
#else
  if (last) {
#endif
      for (int i = 0; i < layers.size(); ++i) {
        if (IsSkipWaitGradient(i)) {
#ifdef FW_OVERLAP_OPT
          finished_count++;
          layer_finished_flags_[i] = true;
#endif
          continue;
        }
#ifdef FW_OVERLAP_OPT
        if (layer_finished_flags_[i])
          continue;
#endif

        WaitAndUpdateGradient(i);
#ifdef FW_OVERLAP_OPT
        if (layer_finished_flags_[i])
          finished_count++;
#endif
      }
#ifdef FW_OVERLAP_OPT
    }
#endif
  }

  DLOG(WARNING) << "iter " << root_solver_->iter() << ", loss " << loss;
  return loss;
}

template <typename Dtype>
Dtype MultiSolver<Dtype>::ForwardBackward() {
  Dtype loss = 0;
  root_solver_->net()->ClearParamDiffs();
  for (int i = 0; i < iter_size; ++i) {
    loss += ForwardBackwardImpl(
      (i == 0), (i + 1 == iter_size));
  }
  return loss / iter_size;
}

INSTANTIATE_CLASS(MultiSolver);

}  // namespace caffe

#endif /* USE_MLSL */
