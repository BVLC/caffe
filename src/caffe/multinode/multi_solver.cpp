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

template <typename Dtype>
Dtype MultiSolver<Dtype>::ForwardBackwardImpl(bool first, bool last) {

  Dtype loss = 0;
  Net<Dtype>& net = *root_solver_->net();
  const std::vector<shared_ptr<Layer<Dtype>>>& layers{ net.layers() };
  const std::vector<bool>& layer_need_backward{ net.layer_need_backward() };

#ifdef CAFFE_PER_LAYER_TIMINGS
  Timer& timer = root_solver_->timer;
  std::vector<double>& forward_time_per_layer = root_solver_->forward_time_per_layer;
  std::vector<double>& backward_time_per_layer = root_solver_->backward_time_per_layer;
  std::vector<double>& update_time_per_layer = root_solver_->update_time_per_layer;
#endif /* CAFFE_PER_LAYER_TIMINGS */

  net.ClearParamDiffs();

  for (int i = 0; i < layers.size(); ++i) {
#ifdef CAFFE_PER_LAYER_TIMINGS
    timer.Start();
#endif
    loss += net.ForwardFromTo(i, i);

#ifdef CAFFE_PER_LAYER_TIMINGS
    forward_time_per_layer[i] += timer.MicroSeconds();
#endif
  }

  for (int i = layers.size() - 1; i >= 0; --i) {
#ifdef CAFFE_PER_LAYER_TIMINGS
    timer.Start();
#endif

    if (!layer_need_backward[i]) {
      continue;
    }

    net.BackwardFromTo(i, i);

    if (last && (layers[i]->layerOp != nullptr) && layers[i]->layerOp->HasParameterSets()) {
      for (int j = 0; j < callbacks_.size(); ++j) {
          callbacks_[j]->on_iter_finished(i);
      }
    }

#ifdef CAFFE_PER_LAYER_TIMINGS
    backward_time_per_layer[i] += timer.MicroSeconds();
#endif
  }

  if (last) {

    for (int i = 0; i < layers.size(); ++i) {
#ifdef CAFFE_PER_LAYER_TIMINGS
      timer.Start();
#endif
      if (!layer_need_backward[i] || ((layers[i]->layerOp != nullptr) && !layers[i]->layerOp->HasParameterSets())) {
        DLOG(INFO) << "ForwardBackwardImpl: no need for apply_updates for layer # " << i
          << ", skip on_delwt_wait, apply_updates, on_wtinc_ready";
        continue;
      }

      for (int j = 0; j < callbacks_.size(); ++j) {
        callbacks_[j]->on_delwt_wait(i);
      }

      for (int j = 0; j < callbacks_.size(); ++j) {
        callbacks_[j]->apply_updates(i);
      }
#ifdef CAFFE_PER_LAYER_TIMINGS
      update_time_per_layer[i] += timer.MicroSeconds();
#endif
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
