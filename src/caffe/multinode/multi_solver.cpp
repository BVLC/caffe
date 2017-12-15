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
#include "caffe/util/performance.hpp"

namespace caffe {

#ifdef CAFFE_PER_LAYER_TIMINGS

#define LAYER_TIMING_START(name, index) do { \
  if (root_solver_->net()->phase() == TRAIN) { \
    root_solver_->net()->name##_start_time_per_layer[index] = root_solver_->net()->timer.Duration(); \
  } \
}while(0)

#define LAYER_TIMING_STOP_2(layer_time_name, name, index) do { \
  if (root_solver_->net()->phase() == TRAIN) { \
    root_solver_->net()->name##_stop_time_per_layer[index] = root_solver_->net()->timer.Duration(); \
    root_solver_->net()->layer_time_name##_time_per_layer[index] += (root_solver_->net()->name##_stop_time_per_layer[index] - root_solver_->net()->name##_start_time_per_layer[index]); \
  } \
}while(0)

#define LAYER_TIMING_STOP(name, index) LAYER_TIMING_STOP_2(name,name,index)

#ifdef FW_OVERLAP_OPT
#define LAYER_WAIT_TIMING_START(index) do { \
  if (root_solver_->net()->phase() == TRAIN) { \
    root_solver_->net()->first_waitcomm_start_time_per_layer[index] = root_solver_->net()->wait_timer.Duration(); \
  } \
}while(0)

#define LAYER_WAIT_TIMING_STOP(index) do { \
  if (root_solver_->net()->phase() == TRAIN) { \
    root_solver_->net()->first_waitcomm_stop_time_per_layer[index] = root_solver_->net()->wait_timer.Duration(); \
    root_solver_->net()->waitcomm_time_per_layer[index] += (root_solver_->net()->first_waitcomm_stop_time_per_layer[index] - root_solver_->net()->first_waitcomm_start_time_per_layer[index]); \
  } \
}while(0)

#define LAYER_REMOVE_UPDATE_TIME(layer_i, layer_k) do { \
  if (root_solver_->net()->phase() == TRAIN) { \
    root_solver_->net()->waitcomm_time_per_layer[layer_i] -= root_solver_->net()->update_time_per_layer[layer_k]; \
  } \
} while (0)
#endif

#else

#define LAYER_TIMING_START(name, index)
#define LAYER_TIMING_STOP(name, index) 
#define LAYER_TIMING_STOP_2(layer_time_name, name,index)

#ifdef FW_OVERLAP_OPT
#define LAYER_WAIT_TIMING_START(index)
#define LAYER_WAIT_TIMING_STOP(index)
#define LAYER_REMOVE_UPDATE_TIME(layer_i, layer_k)
#endif

#endif /* CAFFE_PER_LAYER_TIMINGS */


template <typename Dtype>
inline bool MultiSolver<Dtype>::IsSkipSyncGradient(int layer_id) {
  Net<Dtype>& net = *root_solver_->net();
  const std::vector<shared_ptr<Layer<Dtype>>>& layers{ net.layers() };
  const std::vector<bool>& layer_need_backward{ net.layer_need_backward() };

  if (!layer_need_backward[layer_id] || ((layers[layer_id]->layerOp != nullptr)
        && !layers[layer_id]->layerOp->HasParameterSets())) {
    DLOG(INFO) << "No need for synchronizing gradients for layer # " << layer_id;
    return true;
  }
  return false;
}

template <typename Dtype>
inline bool MultiSolver<Dtype>::WaitGradient(int layer_id) {
  for (int j = 0; j < callbacks_.size(); ++j) {
    callbacks_[j]->on_delwt_wait(layer_id);
  }

#ifdef FW_OVERLAP_OPT
  return layer_finished_flags_[layer_id];
#else
  return true;
#endif
}

template <typename Dtype>
inline void MultiSolver<Dtype>::UpdateGradient(int layer_id) {
#ifdef FW_OVERLAP_OPT
    CHECK(layer_finished_flags_[layer_id]);
#endif
    PERFORMANCE_MEASUREMENT_BEGIN();
    for (int j = 0; j < callbacks_.size(); ++j) {
      callbacks_[j]->apply_updates(layer_id);
    }
    PERFORMANCE_MEASUREMENT_END_STATIC("weights_update");
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
      LAYER_WAIT_TIMING_START(i);
      while (layer_finished_flags_[i] == false) {
        if (IsSkipSyncGradient(i))
          break;
        if (WaitGradient(i)) {
          // The function call cannot be moved out of while loop. Otherwise,
          // at first iteration, additional UpdateGradient will be called,
          // even if no gradient is synced.
          LAYER_TIMING_START(first_update, i);
          UpdateGradient(i);
          LAYER_TIMING_STOP_2(update, first_update, i);

          // The update time for layer i must be removed from waitcomm time
          // for layer i
          LAYER_REMOVE_UPDATE_TIME(i, i);
          break;
        }

        // wait and update gradient for next layers
        for (int k=i+1; k<layers.size(); k++) {
          if (layer_finished_flags_[k] || IsSkipSyncGradient(k)) {
            layer_finished_flags_[k] = true;
            continue;
          }
          if (WaitGradient(k)) {
            LAYER_TIMING_START(first_update, k);
            UpdateGradient(k);
            LAYER_TIMING_STOP_2(update, first_update, k);

            // The update time for layer k must be removed from waitcomm time
            // for layer i
            LAYER_REMOVE_UPDATE_TIME(i, k);
            break;
          }
        }
      }
      LAYER_WAIT_TIMING_STOP(i);
      // set flag to false after updating gradient
      layer_finished_flags_[i] = false;
    }
#endif

    loss += net.ForwardFromTo(i, i);
  }
  
  // Clear parameter diffs after communication is finished (that is, after 
  // calling WaitGradientComm)
  if (first) {
    root_solver_->net()->ClearParamDiffs();
  }

  for (int i = layers.size() - 1; i >= 0; --i) {
    if (!layer_need_backward[i]) {
      continue;
    }
    
    net.BackwardFromTo(i, i);

    LAYER_TIMING_START(startcomm, i);
    if (last && IsSkipSyncGradient(i) == false) {
      for (int j = 0; j < callbacks_.size(); ++j) {
        callbacks_[j]->on_backward_finished(i);
      }
    }
    LAYER_TIMING_STOP(startcomm, i);
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
      LAYER_TIMING_START(waitcomm, i);
      if (IsSkipSyncGradient(i)) {
        LAYER_TIMING_STOP(waitcomm, i);
        continue;
      }

#ifdef FW_OVERLAP_OPT
      while (
#endif
        WaitGradient(i)
#ifdef FW_OVERLAP_OPT
          == false)
#endif
      ;
      LAYER_TIMING_STOP(waitcomm, i);

      LAYER_TIMING_START(update, i);
      UpdateGradient(i);
      LAYER_TIMING_STOP(update, i);
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
