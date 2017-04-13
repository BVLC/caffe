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


#ifndef CAFFE_MLSLSYNC_HPP_
#define CAFFE_MLSLSYNC_HPP_

#include <string>
#include "caffe/solver.hpp"

#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <glog/logging.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/multinode/multi_sync.hpp"
#include "caffe/multinode/multi_solver.hpp"

#include "mlsl.h"

using namespace MLSL;

namespace caffe {

#define CAN_USE_PRV(param) (0) //(param->prv_diff() && (param->prv_diff_count() == param->count()))

template <typename Dtype>
class MlslSync : public MlslSolver<Dtype>::Callback {

    shared_ptr<MlslSolver<Dtype> > solver;
    bool initialized;
    boost::thread::id solver_thread_id;
    int snapshot_per_iters;

    vector<shared_ptr<Layer<Dtype> > > layers;
    shared_ptr<Net<Dtype> > net;
    const vector<Blob<Dtype>*>& net_params;
    vector<vector<int> > layer_param_ids;

    vector<vector<int> > bottom_pack_block_nums;
    vector<vector<int> > bottom_unpack_block_nums;
    vector<vector<int> > top_pack_block_nums;
    vector<vector<int> > top_unpack_block_nums;

    bool is_root; // MLSL::GetNodeId() == 0

public:

    MlslSync(shared_ptr<Solver<Dtype> >);
    ~MlslSync();

    void snapshot() {
        if (is_root) {
            for (int layer_id = 0; layer_id < layers.size(); ++layer_id) {
                //apply_updates(layer_id);
            }
            solver->root_solver()->Snapshot();
        }
    }
    
    void synchronize_params() {
        // FIXME: use MLSL API to bcast initial weights values
        LOG(WARNING) << "synchronize_params: bcast";
        for (int idx = 0; idx < net_params.size(); ++idx) {
            MPI_Bcast(net_params[idx]->mutable_cpu_data(),
                      net_params[idx]->count(),
                      (sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE,
                      0,
                      MPI_COMM_WORLD);
        }
    }

    void run() {
        LOG(WARNING) << "RUN: "
                     << "PER LAYER TIMINGS ARE"
#ifdef CAFFE_PER_LAYER_TIMINGS
                     << " ENABLED"
#else
                     << " DISABLED"
#endif
                     << ", SINGLE DB SPLITTING IS"
#ifdef CAFFE_MLSL_SHUFFLE
                     << " ENABLED"
#else
                     << " DISABLED"
#endif
                     ;


        synchronize_params();
                                                                                                                                    
        solver->add_callback(this);
        solver->Solve();
        if (is_root) {
            //solver->root_solver()->Snapshot();
        }
  }

  void set_solver_thread() {
      solver_thread_id = boost::this_thread::get_id();
  }

  void check_snapshot() {
      CHECK(boost::this_thread::get_id() == solver_thread_id);
      if (!is_root) return;

      if ((snapshot_per_iters != 0)
          && (solver->root_solver()->iter() % snapshot_per_iters == 0)) {
          solver->root_solver()->Snapshot();
      }
  }

  void apply_updates(int layer_id) {

      CHECK(boost::this_thread::get_id() == solver_thread_id);

      shared_ptr<Layer<Dtype> > layer = layers[layer_id];
      vector<int>& param_ids = layer_param_ids[layer_id];
      LOG_LAYER(layer) << "bprop: apply_updates: layer_id " << layer_id << ", param_ids size " << param_ids.size();

      for (int i = 0; i < param_ids.size(); ++i) {
          LOG_BLOB(layer, net_params[param_ids[i]], diff, param_ids[i], "bprop: apply_updates: delwt for sgd:");
          solver->root_solver()->ApplyUpdate(param_ids[i]);
          LOG_BLOB(layer, net_params[param_ids[i]], diff, param_ids[i], "bprop: apply_updates: wtinc after sgd:");
      }
  }

  void on_start() {
      if (!initialized) {
          set_solver_thread();
          initialized = true;
      }
      check_snapshot();
      DLOG(INFO) << "started iteration " << solver->root_solver()->iter();
  }



  // main callback for MlslSolver loop
  void on_forward_start(int layer_id) {
      shared_ptr<Layer<Dtype> > layer = layers[layer_id];
      int bottom_size = layer->layer_param().bottom_size();
      LOG_LAYER(layer) << "fprop: on_forward_start: layer_id " << layer_id << ", bottom_size " << bottom_size;

      for (int bottom_id = 0; bottom_id < bottom_size; ++bottom_id) {

          if (!bottom_unpack_block_nums[layer_id][bottom_id]) {
              LOG_LAYER(layer) << "fprop: on_forward_start: skip CommsWait for bottom_id " << bottom_id;
              continue;
          }
          
          FeatureMap *fm = layer->layerOp->InputFeatureMap(bottom_id);
          LOG_LAYER(layer) << "fprop: on_forward_start: wait data from bottom_id " << bottom_id;
          Dtype *comms_buf = (Dtype *)fm->CommsWait();
          LOG_LAYER(layer) << "fprop: on_forward_start: got data from bottom_id " << bottom_id;

          if (comms_buf) {
              layer->unpack_buffer(fm, comms_buf, layer->bottom_vec[bottom_id]->mutable_cpu_data());
              LOG_BLOB(layer, layer->bottom_vec[bottom_id], data, bottom_id, "fprop: on_forward_start: bottom_data:");
              LOG_BUFFER(layer, comms_buf, bottom_id, "fprop: on_forward_start: comms_buf:");
          }
      }

#ifdef DEBUG
      if (layer->layerOp->HasWeights()) {
          vector<int>& param_ids = layer_param_ids[layer_id];
          LOG_LAYER(layer) << "fprop: on_forward_start: param_ids size " << param_ids.size();

          for (int i = 0; i < param_ids.size(); ++i) {
              LOG_BLOB(layer, net_params[param_ids[i]], data, param_ids[i], "fprop: on_forward_start: weigths_data:");
          }
      }
#endif

  }

  void on_forward_finished(int layer_id) {

      shared_ptr<Layer<Dtype> > layer = layers[layer_id];
      int top_size = layer->layer_param().top_size();
      LOG_LAYER(layer) << "fprop: on_forward_finished: layer_id " << layer_id << ", top_size " << top_size;

      for (int top_id = 0; top_id < top_size; ++top_id) {

          if (!top_pack_block_nums[layer_id][top_id]) {
              LOG_LAYER(layer) << "fprop: on_forward_finished: skip CommsStart for top_id " << top_id;
              continue;
          }
          
          FeatureMap *fm = layer->layerOp->OutputFeatureMap(top_id);
          Dtype* comms_buf = (Dtype *)fm->CBuf()->GetPtr();

          if (comms_buf) {
              layer->pack_buffer(fm, comms_buf, layer->top_vec[top_id]->cpu_data());
              LOG_BLOB(layer, layer->top_vec[top_id], data, top_id, "fprop: on_forward_finished: top_data:");
              LOG_BUFFER(layer, comms_buf, top_id, "fprop: on_forward_finished: comms_buf:");
              LOG_LAYER(layer) << "fprop: on_forward_finished: send data to top_id " << top_id;
              fm->CommsStart(comms_buf);
          }
      }

#ifdef DEBUG
      if (layer->layerOp->HasWeights()) {
          vector<int>& param_ids = layer_param_ids[layer_id];
          LOG_LAYER(layer) << "fprop: on_forward_finished: param_ids size " << param_ids.size();

          for (int i = 0; i < param_ids.size(); ++i) {
              LOG_BLOB(layer, net_params[param_ids[i]], data, param_ids[i], "fprop: on_forward_finished: weigths_data:");
          }
      }
#endif

  }

  void on_backward_start(int layer_id) {
      shared_ptr<Layer<Dtype> > layer = layers[layer_id];
      int top_size = layer->layer_param().top_size();
      LOG_LAYER(layer) << "bprop: on_backward_start: layer_id " << layer_id << ", top size " << top_size;

      for (int top_id = 0; top_id < top_size; ++top_id) {

          if (!top_unpack_block_nums[layer_id][top_id]) {
              LOG_LAYER(layer) << "bprop: on_backward_start: skip CommsWait for top_id " << top_id;
              continue;
          }

          FeatureMap *fm = layer->layerOp->OutputFeatureMap(top_id);
          Dtype *comms_buf = (Dtype *)fm->CommsWait();
          LOG_LAYER(layer) << "bprop: on_backward_start: got delout from top_id " << top_id;

          if (comms_buf) {
              layer->unpack_buffer(fm, comms_buf, layer->top_vec[top_id]->mutable_cpu_diff());
              LOG_BLOB(layer, layer->top_vec[top_id], diff, top_id, "bprop: on_backward_start: top_diff:");
              LOG_BUFFER(layer, comms_buf, top_id, "bprop: on_backward_start: comms_buf:");
          }
      }
  }

  void on_iter_finished(int layer_id) {
      shared_ptr<Layer<Dtype> > layer = layers[layer_id];
      LOG_LAYER(layer) << "bprop: on_iter_finished: iter " << solver->root_solver()->iter() << ", layer id " << layer_id;

      vector<int>& param_ids = layer_param_ids[layer_id];
      LOG_LAYER(layer) << "bprop: on_iter_finished: param_ids size " << param_ids.size();

      for (int i = 0; i < param_ids.size(); ++i) {

          LOG_LAYER(layer) << "bprop: on_iter_finished: start delwt for param_id " << param_ids[i];
          LOG_BLOB(layer, net_params[param_ids[i]], diff, param_ids[i], "bprop: on_iter_finished: delwt:");
          
          if (CAN_USE_PRV(net_params[param_ids[i]]))
              layer->layerOp->GetWeights(i)->CommsStartDelWt((void*)net_params[param_ids[i]]->mutable_prv_diff());
          else
              layer->layerOp->GetWeights(i)->CommsStartDelWt((void*)net_params[param_ids[i]]->mutable_cpu_diff());

          LOG_BLOB(layer, net_params[param_ids[i]], diff, param_ids[i], "bprop: on_iter_finished:  delwt before comms:");
      }
  }

  void on_delwt_wait(int layer_id) {
      shared_ptr<Layer<Dtype> > layer = layers[layer_id];
      LOG_LAYER(layer) << "bprop: on_delwt_wait: iter " << solver->root_solver()->iter() << ", layer id " << layer_id;

      vector<int>& param_ids = layer_param_ids[layer_id];
      LOG_LAYER(layer) << "bprop: on_delwt_wait: param_ids size " << param_ids.size();

      for (int i = 0; i < param_ids.size(); ++i) {

          LOG_LAYER(layer) << "bprop: on_delwt_wait: wait delwt for param_id " << param_ids[i];
          Dtype* delwt_buf = (Dtype*)layer->layerOp->GetWeights(i)->CommsWaitDelWt();
          LOG_LAYER(layer) << "bprop: on_delwt_wait: got delwt for param_id " << param_ids[i];

          LOG_BLOB(layer, net_params[param_ids[i]], diff, param_ids[i], "bprop: on_delwt_wait: delwt after comms:");
          LOG_BUFFER(layer, delwt_buf, param_ids[i], "bprop: on_delwt_wait: comms buffer:");

          if (delwt_buf)
          {
              if (CAN_USE_PRV(net_params[param_ids[i]])) {
                  if (delwt_buf != net_params[param_ids[i]]->prv_diff())
                      caffe_copy(net_params[param_ids[i]]->count(),
                                 delwt_buf,
                                 net_params[param_ids[i]]->mutable_prv_diff());
              }
              else if (delwt_buf != net_params[param_ids[i]]->cpu_diff())
                  caffe_copy(net_params[param_ids[i]]->count(),
                             delwt_buf,
                             net_params[param_ids[i]]->mutable_cpu_diff());

          }

          LOG_LAYER(layer) << "bprop: on_delwt_wait: got delwt for param_id " << param_ids[i];
          LOG_BLOB(layer, net_params[param_ids[i]], diff, param_ids[i], "bprop: on_delwt_wait: delwt:");
      }
  }

  void on_gradients_ready() {
      DLOG(INFO) << "finished iteration " << solver->root_solver()->iter();
  }
  
};

} // namespace caffe


#endif  // CAFFE_MLSLSYNC_HPP_

#endif /* USE_MLSL */

