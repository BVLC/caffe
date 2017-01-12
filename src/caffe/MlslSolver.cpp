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

#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <vector>
#include "caffe/internode/tree_cluster.hpp"
#include "caffe/MlslSolver.hpp"

namespace caffe {

template <typename Dtype>
MlslSolver<Dtype>::MlslSolver(shared_ptr<Solver<Dtype> > root_solver)
  : root_solver_(root_solver)
  , iter_size(root_solver->param().iter_size())
  , multi_node(MLSL::GetNumNodes() > 1) {

  root_solver->set_forward_backward(
    boost::bind(&MlslSolver<Dtype>::ForwardBackward, this));
}

template <typename Dtype>
Dtype MlslSolver<Dtype>::ForwardBackwardImpl(bool first, bool last) {

  Dtype loss = 0;
  Net<Dtype>& net = *root_solver_->net();
  const vector<shared_ptr<Layer<Dtype> > >& layers = net.layers();
  const vector<bool>& layer_need_backward = net.layer_need_backward();


#ifdef CAFFE_PER_LAYER_TIMINGS
  Timer& timer = root_solver_->timer;
  std::vector<double>& forward_time_per_layer = root_solver_->forward_time_per_layer;
  std::vector<double>& backward_time_per_layer = root_solver_->backward_time_per_layer;
  std::vector<double>& update_time_per_layer = root_solver_->update_time_per_layer;
#endif /* CAFFE_PER_LAYER_TIMINGS */

#ifndef DISTR_WEIGHT_UPDATE
  net.ClearParamDiffs();
#endif

  for (int i = 0; i < layers.size(); ++i) {

#ifdef CAFFE_PER_LAYER_TIMINGS
    timer.Start();
#endif

#ifdef DISTR_WEIGHT_UPDATE
    if (first && layers[i]->layerOp->HasWeights()) {
      for (int j = 0; j < callbacks_.size(); ++j) {
        callbacks_[j]->on_iter_start(i); // wait wtinc
      }
    }
#endif

    if (multi_node && layers[i]->layerOp->NumInputFeatureMaps()) {
        for (int j = 0; j < callbacks_.size(); ++j) {
            callbacks_[j]->on_forward_start(i); // wait input
        }
    }

    shared_ptr<Layer<Dtype> > layer = net.layers()[i];

    for (int bottom_idx = 0; bottom_idx < net.bottom_vecs()[i].size(); bottom_idx++)
        LOG_BLOB(layer, net.bottom_vecs()[i][bottom_idx], data, bottom_idx, "fprop: input data values:");

    for (int param_idx = 0; param_idx < layer->blobs().size(); param_idx++) {
          LOG_BLOB(layer, layer->blobs()[param_idx], data, param_idx, "fprop: weights:");
    }

    Dtype layer_loss = net.ForwardFromTo(i, i);

    for (int top_idx = 0; top_idx < net.top_vecs()[i].size(); top_idx++)
        LOG_BLOB(layer, net.top_vecs()[i][top_idx], data, top_idx, "fprop: output data values:");


    loss += layer_loss;

    DLOG(WARNING) << "iter " << root_solver_->iter() 
                  << ", layer_id " << i
                  << ", layer_type " << layers[i]->type()
                  << ", layer_loss " << layer_loss;

    if (multi_node && layers[i]->layerOp->NumOutputFeatureMaps()) {
        for (int j = 0; j < callbacks_.size(); ++j) {
          callbacks_[j]->on_forward_finished(i); // start ouput
        }
    }

#ifdef CAFFE_PER_LAYER_TIMINGS
    forward_time_per_layer[i] += timer.MicroSeconds();
#endif

  }


  for (int i = layers.size() - 1; i >= 0; --i) {

#ifdef CAFFE_PER_LAYER_TIMINGS
    timer.Start();
#endif

    if (!layer_need_backward[i]) {
      DLOG(INFO) << "ForwardBackwardImpl: no need for backprop for layer # " << i << ", skip on_bprop_start, bprop, on_delinp_ready";
      continue;
    }

    if (multi_node && layers[i]->layerOp->NumOutputFeatureMaps()) {
        for (int j = 0; j < callbacks_.size(); ++j) {
            callbacks_[j]->on_backward_start(i); // wait delout
        }
    }
    
    shared_ptr<Layer<Dtype> > layer = net.layers()[i];

    for (int top_idx = 0; top_idx < net.top_vecs()[i].size(); top_idx++)
        LOG_BLOB(layer, net.top_vecs()[i][top_idx], diff, top_idx, "bprop: input diff values:");


    net.BackwardFromTo(i, i); // start delinp in the middle of bprop if layer has weights (for overlapping with delwt calculation)


    for (int bottom_idx = 0; bottom_idx < net.bottom_vecs()[i].size(); bottom_idx++)
        LOG_BLOB(layer, net.bottom_vecs()[i][bottom_idx], diff, bottom_idx, "bprop: output diff values:");

    if (multi_node && !layers[i]->layerOp->HasWeights() && layers[i]->layerOp->NumInputFeatureMaps()) // otherwise start delinp here
        layers[i]->on_delinp_ready(net.bottom_need_backward()[i]);

    if (multi_node && last && layers[i]->layerOp->HasWeights()) {
      for (int j = 0; j < callbacks_.size(); ++j) {
          callbacks_[j]->on_iter_finished(i); // start delwt
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

      if (!layer_need_backward[i] || !layers[i]->layerOp->HasWeights()) {
        DLOG(INFO) << "ForwardBackwardImpl: no need for apply_updates for layer # " << i
                   << ", skip on_delwt_wait, apply_updates, on_wtinc_ready";
        continue;
      }

      if (multi_node) {
          for (int j = 0; j < callbacks_.size(); ++j) {
              callbacks_[j]->on_delwt_wait(i);
          }
      }

      shared_ptr<Layer<Dtype> > layer = net.layers()[i];

      for (int param_idx = 0; param_idx < layer->blobs().size(); param_idx++) {
          LOG_BLOB(layer, layer->blobs()[param_idx], diff, param_idx, "bprop: delwt:");
      }

      for (int j = 0; j < callbacks_.size(); ++j) {
          callbacks_[j]->apply_updates(i);
      }

#ifdef DISTR_WEIGHT_UPDATE
      for (int j = 0; j < callbacks_.size(); ++j) {
          callbacks_[j]->on_wtinc_ready(i);
      }
      
      if (root_solver_->iter() == (root_solver_->param().max_iter() - 1)) // it is the last iter at all, apply updates before exiting
      {
          for (int j = 0; j < callbacks_.size(); ++j) {
              callbacks_[j]->on_iter_start(i); // wait wtinc
          }
      }

#endif

#ifdef CAFFE_PER_LAYER_TIMINGS
      update_time_per_layer[i] += timer.MicroSeconds();
#endif

    }
  }

#ifndef DISTR_WEIGHT_UPDATE
  // FIXME: we should sync params about once in epoch, currently 830 for 1536 batchsize
  /*for (int j = 0; j < callbacks_.size(); ++j) {
      if (root_solver_->iter() % 830 == 0)
          callbacks_[j]->synchronize_params();
  }*/
#endif

  DLOG(WARNING) << "iter " << root_solver_->iter() << ", loss " << loss;
  return loss;
}

template <typename Dtype>
Dtype MlslSolver<Dtype>::ForwardBackward() {
  Dtype loss = 0;
  for (int i = 0; i < iter_size; ++i) {
    loss += ForwardBackwardImpl(
      (i == 0), (i + 1 == iter_size));
  }
  return loss / iter_size;
}

template <typename Dtype>
void MlslSolver<Dtype>::Solve() {
  root_solver_->Solve();
}

INSTANTIATE_CLASS(MlslSolver);

}  // namespace caffe

#endif /* USE_MLSL */
