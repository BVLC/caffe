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

#ifndef CAFFE_MULTISYNC_HPP_
#define CAFFE_MULTISYNC_HPP_

#ifdef USE_MLSL

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
#include "caffe/multinode/mlsl.hpp"
#include "caffe/multinode/multi_solver.hpp"

namespace caffe {

#define CAN_USE_PRV(param) false //(param->prv_diff() && (param->prv_diff_count() == param->count()))

  template <typename Dtype>
  class MultiSync : public MultiSolver<Dtype>::Callback {

    boost::shared_ptr<MultiSolver<Dtype>> solver;
    int snapshot_per_iters;

    vector<shared_ptr<Layer<Dtype>>> layers;
    shared_ptr<Net<Dtype>> net;
    const vector<Blob<Dtype> *> &net_params;
    vector<vector<int>> layer_param_ids;

    vector<vector<int>> bottom_pack_block_nums;
    vector<vector<int>> bottom_unpack_block_nums;
    vector<vector<int>> top_pack_block_nums;
    vector<vector<int>> top_unpack_block_nums;

    bool is_root; // MLSL::GetNodeId() == 0

  public:

    MultiSync(shared_ptr<Solver<Dtype> >);

    ~MultiSync();

    void snapshot() {
      if (is_root) {
        for (int layer_id = 0; layer_id < layers.size(); ++layer_id) {
          //apply_updates(layer_id);
        }
        solver->root_solver()->Snapshot();
      }
    }

    void synchronize_parameters() {
      LOG(WARNING) << "synchronize_params: bcast";
      for (int idx = 0; idx < net_params.size(); ++idx) {
        mn::bcast(net_params[idx]->mutable_cpu_data(), net_params[idx]->count());
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

      synchronize_parameters();
      mn::train::commit();
      solver->add_callback(this);
      solver->Solve();
      if (is_root) {
        //solver->root_solver()->Snapshot();
      }
    }

    void check_snapshot() {
      if (is_root) {
        if ((snapshot_per_iters != 0) && (solver->root_solver()->iter() % snapshot_per_iters == 0)) {
          solver->root_solver()->Snapshot();
        }
      }
    }

    void apply_updates(int layer_id) {
      std::vector<int> &param_ids = layer_param_ids[layer_id];
      for (int i = 0; i < param_ids.size(); ++i) {
        solver->root_solver()->ApplyUpdate(param_ids[i]);
      }
    }

    void on_start() {
      check_snapshot();
      DLOG(INFO) << "started iteration " << solver->root_solver()->iter();
    }

    // main callback for MultiSolver loop
    void on_forward_start(int layer_id) {
      boost::shared_ptr<Layer<Dtype>> &layer = layers[layer_id];
      int bottom_size{layer->layer_param().bottom_size()};
      for (int i = 0; i < bottom_size; ++i) {
        if (bottom_unpack_block_nums[layer_id][i] != 0) {
          MLSL::Activation *activation{layer->layerOp->GetInput(i)};
          Dtype *comms_buf{(Dtype *) activation->WaitComm()};
          if (comms_buf != nullptr) {
            layer->unpack_buffer(activation, comms_buf, layer->bottom_vec[i]->mutable_cpu_data());
          }
        }
      }
    }

    void on_forward_finished(int layer_id) {
      boost::shared_ptr<Layer<Dtype>> &layer = layers[layer_id];
      int top_size{layer->layer_param().top_size()};
      for (int i = 0; i < top_size; ++i) {
        if (top_pack_block_nums[layer_id][i] != 0) {
          MLSL::Activation *activation{layer->layerOp->GetOutput(i)};
          Dtype *comms_buf{(Dtype *) activation->GetCommBuf()};
          if (comms_buf) {
            layer->pack_buffer(activation, comms_buf, layer->top_vec[i]->cpu_data());
            activation->StartComm(comms_buf);
          }
        }
      }
    }

    void on_backward_start(int layer_id) {
      boost::shared_ptr<Layer<Dtype>> &layer = layers[layer_id];
      int top_size{layer->layer_param().top_size()};
      for (int i = 0; i < top_size; ++i) {
        if (top_unpack_block_nums[layer_id][i] != 0) {
          MLSL::Activation *activation{layer->layerOp->GetOutput(i)};
          Dtype *comms_buf{(Dtype *) activation->WaitComm()};
          if (comms_buf) {
            layer->unpack_buffer(activation, comms_buf, layer->top_vec[i]->mutable_cpu_diff());
          }
        }
      }
    }

    void on_iter_finished(int layer_id) {
      boost::shared_ptr<Layer<Dtype>> &layer = layers[layer_id];
      std::vector<int> &param_ids = layer_param_ids[layer_id];
      for (int i = 0; i < param_ids.size(); ++i) {
        if (CAN_USE_PRV(net_params[param_ids[i]])) {
          layer->layerOp->GetParameterSet(i)->StartGradientComm((void *) net_params[param_ids[i]]->mutable_prv_diff());
        } else {
          layer->layerOp->GetParameterSet(i)->StartGradientComm((void *) net_params[param_ids[i]]->mutable_cpu_diff());
        }
      }
    }

    void on_delwt_wait(int layer_id) {
      boost::shared_ptr<Layer<Dtype>> &layer = layers[layer_id];
      std::vector<int> &param_ids = layer_param_ids[layer_id];

      for (int i = 0; i < param_ids.size(); ++i) {
        Dtype *delwt_buf{(Dtype *) layer->layerOp->GetParameterSet(i)->WaitGradientComm()};
        if (delwt_buf) {
          if (CAN_USE_PRV(net_params[param_ids[i]])) {
            if (delwt_buf != net_params[param_ids[i]]->prv_diff())
              caffe_copy(net_params[param_ids[i]]->count(),
                         delwt_buf,
                         net_params[param_ids[i]]->mutable_prv_diff());
          } else if (delwt_buf != net_params[param_ids[i]]->cpu_diff())
            caffe_copy(net_params[param_ids[i]]->count(),
                       delwt_buf,
                       net_params[param_ids[i]]->mutable_cpu_diff());

        }
      }
    }

    void on_gradients_ready() {
      DLOG(INFO) << "finished iteration " << solver->root_solver()->iter();
    }

  };

} // namespace caffe

#endif /* USE_MLSL */

#endif  // CAFFE_MULTISYNC_HPP_
