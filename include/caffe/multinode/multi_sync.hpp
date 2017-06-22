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

    vector<shared_ptr<Layer<Dtype>>> layers;
    shared_ptr<Net<Dtype>> net;
    const vector<Blob<Dtype> *> &net_params;
    vector<vector<int>> layer_param_ids;
    // layer_id -> blob_id -> cached blob to restore
    // statistics
    vector<vector<shared_ptr<Blob<Dtype>>>> cached_stats;

#ifdef PERFORMANCE_MONITORING
    #define STATS_OUTPUT_FILE "mlsl_stats.txt"

    struct StatsIterResult {
        unsigned long long isolationCommTime;
        unsigned long long commTime;
        unsigned long long computeTime;
        size_t commSize;
    };

    // Operations[Iteration]
    vector<vector<StatsIterResult>> statsIterResult;

    unsigned long long totalIsolationCommTime;
    unsigned long long totalCommTime;
    unsigned long long totalComputeTime;
    size_t totalCommSize;
#endif

  public:

    MultiSync(shared_ptr<Solver<Dtype> >);

    virtual ~MultiSync() {
    }

    void synchronize_parameters() {
      LOG(INFO) << "synchronize_params: bcast";
      for (int i = 0; i < layers.size(); i++) {
        mn::Distribution &distrib = layers[i]->GetDistribution();
        for (int j = 0; j < layer_param_ids[i].size(); j++) {
          int layer_param_id = layer_param_ids[i][j];
          distrib.bcast<Dtype,MLSL::GT_DATA>(
            net_params[layer_param_id]->mutable_cpu_data(),
            net_params[layer_param_id]->count());
        }
      }
    }

    void synchronize_statistics() {
      cached_stats.resize(layers.size());
      for (int i = 0; i < layers.size(); i++) {
        if (string(layers[i]->type()) == "BatchNorm" &&
            !layers[i]->layer_param().batch_norm_param().use_global_stats()) {
          vector<shared_ptr<Blob<Dtype>>> cached_blobs;
          // 3 blobs: mean, variance and scaling factor
          for (int j = 0; j < layer_param_ids[i].size() && j < 3; j++) {
            shared_ptr<Blob<Dtype>> b = shared_ptr<Blob<Dtype>>(new Blob<Dtype>());
            Blob<Dtype> *net_param = net_params[layer_param_ids[i][j]];
            b->ReshapeLike(*net_param);
            b->CopyFrom(*net_param);
            cached_blobs.push_back(b);
            mn::Distribution &distrib = layers[i]->GetDistribution();
            distrib.allreduce<Dtype,MLSL::RT_SUM,MLSL::GT_DATA>(
              net_param->mutable_cpu_data(), net_param->mutable_cpu_data(),
              net_param->count());
          }
          cached_stats[i] = cached_blobs;
        }
      }
    }

    void restore_statistics() {
      for (int i = 0; i < layers.size(); i++) {
        if (string(layers[i]->type()) == "BatchNorm" &&
          !layers[i]->layer_param().batch_norm_param().use_global_stats()) {
          // 3 blobs: mean, variance and scaling factor
          for (int j = 0; j < layer_param_ids[i].size() && j < 3; j++) {
            Blob<Dtype> *net_param = net_params[layer_param_ids[i][j]];
            net_param->CopyFrom(*cached_stats[i][j]);
          }
        }
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
                   << " ENABLED";
#else
                   << " DISABLED";
#endif

      synchronize_parameters();
      mn::train::commit();

#ifdef PERFORMANCE_MONITORING
  statsIterResult.resize(caffe::mn::train::get_session().get_operation_count());
  caffe::mn::train::stats::start();
#endif

      solver->add_callback(this);
      solver->Solve();

#ifdef PERFORMANCE_MONITORING
    dump_stats_to_file();
#endif
    }

    void apply_updates(int layer_id) {
      std::vector<int> &param_ids = layer_param_ids[layer_id];
      for (int i = 0; i < param_ids.size(); ++i) {
        solver->root_solver()->ApplyUpdate(param_ids[i]);
      }
    }

    void on_start() {
      DLOG(INFO) << "started iteration " << solver->root_solver()->iter();
    }

    void on_iter_finished(int layer_id) {
      boost::shared_ptr<Layer<Dtype>> &layer = layers[layer_id];
      if (layer->layerOp == nullptr) {
        return;
      }

      std::vector<int> &param_ids = layer_param_ids[layer_id];
      for (int i = 0; i < param_ids.size(); ++i) {
        if (!layer->ParamNeedReduce(param_ids[i])) continue;
        if (CAN_USE_PRV(net_params[param_ids[i]])) {
          layer->layerOp->GetParameterSet(i)->StartGradientComm((void *) net_params[param_ids[i]]->mutable_prv_diff());
        } else {
          layer->layerOp->GetParameterSet(i)->StartGradientComm((void *) net_params[param_ids[i]]->mutable_cpu_diff());
        }
      }
    }

    void on_delwt_wait(int layer_id) {
      boost::shared_ptr<Layer<Dtype>> &layer = layers[layer_id];
      if (layer->layerOp == nullptr) {
        return;
      }

      std::vector<int> &param_ids = layer_param_ids[layer_id];

      for (int i=0; i<param_ids.size(); i++) {
        if (!layer->ParamNeedReduce(param_ids[i])) continue;
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

#ifdef PERFORMANCE_MONITORING
      caffe::mn::train::stats::stop();

      size_t opCount = caffe::mn::train::get_session().get_operation_count();

      for (size_t opIdx = 0; opIdx < opCount; ++opIdx) {
          StatsIterResult iterResult;

          iterResult.isolationCommTime = caffe::mn::train::stats::get_isolation_comm_time(opIdx);
          iterResult.commTime = caffe::mn::train::stats::get_comm_time(opIdx);
          iterResult.computeTime = caffe::mn::train::stats::get_compute_time(opIdx);
          iterResult.commSize = caffe::mn::train::stats::get_comm_size(opIdx);

          statsIterResult[opIdx].push_back(iterResult);

          // Save total values before reset statistics
          totalIsolationCommTime = caffe::mn::train::stats::get_total_isolation_comm_time();
          totalCommTime = caffe::mn::train::stats::get_total_comm_time();
          totalComputeTime = caffe::mn::train::stats::get_total_compute_time();
          totalCommSize = caffe::mn::train::stats::get_total_comm_size();
      }

      caffe::mn::train::stats::reset();
      caffe::mn::train::stats::start();
#endif //PERFORMANCE_MONITORING
    }

    void on_before_test() {
      synchronize_statistics();
      synchronize_parameters();
    }

    void on_after_test() {
      restore_statistics();
    }

    void on_before_snapshot() {
      synchronize_statistics();
    }

    void on_after_snapshot() {
      restore_statistics();
    }

#ifdef PERFORMANCE_MONITORING
    void dump_stats_to_file() {
      FILE* outputFile = fopen(STATS_OUTPUT_FILE, "w");
      if(outputFile == NULL) {
        LOG(ERROR) << "unable to create file " << STATS_OUTPUT_FILE;
        return;
      }

      size_t opCount = caffe::mn::train::get_session().get_operation_count();

      // Write file header
      fprintf(outputFile, "    MLSL common communication statistics\n\n");

      fprintf(outputFile, "Total IsolationCommTime: %12llu\n",  totalIsolationCommTime);
      fprintf(outputFile, "Total CommTime:          %12llu\n",  totalCommTime);
      fprintf(outputFile, "Total ComputeTime:       %12llu\n",  totalComputeTime);
      fprintf(outputFile, "Total CommSize:          %12zu\n",   totalCommSize);
      fprintf(outputFile, "Num operations:          %12zu\n\n", opCount);

      fprintf(outputFile, "    MLSL detailed communication statistics\n\n");

      fprintf(outputFile, "Format:\n");
      fprintf(outputFile, "  OperationName\n");
      fprintf(outputFile, "  Iteration, IsolationCommTime (kCycles), CommTime (kCycles), ComputeTime (kCycles), CommSize (KB)\n");

      // Write all iteratons for each layer
      for (size_t opIdx = 0; opIdx < opCount; ++opIdx) {
        fprintf(outputFile, "\n%s\n\n", caffe::mn::train::get_session().get_operation_name(opIdx));
        for (size_t iter = 0; iter < statsIterResult[opIdx].size(); ++iter) {
          fprintf(outputFile, "%6zu %11llu %11llu %11llu %8zu\n",
                  iter+1,
                  statsIterResult[opIdx][iter].isolationCommTime,
                  statsIterResult[opIdx][iter].commTime,
                  statsIterResult[opIdx][iter].computeTime,
                  statsIterResult[opIdx][iter].commSize);
        }
      }

      fclose(outputFile);
    }
#endif //PERFORMANCE_MONITORING
  };

} // namespace caffe

#endif /* USE_MLSL */

#endif  // CAFFE_MULTISYNC_HPP_
