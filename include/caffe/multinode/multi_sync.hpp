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
#include <list>

#include "caffe/caffe.hpp"
#include "caffe/multinode/mlsl.hpp"
#include "caffe/multinode/multi_solver.hpp"
#include "caffe/multinode/async_param_server.hpp"

namespace caffe {

#define CAN_USE_PRV_DATA(param) (param->prv_data() && (param->prv_data_count() == param->count()))
#define CAN_USE_PRV_DIFF(param) (param->prv_diff() && (param->prv_diff_count() == param->count()))

  struct AsyncTask {
    int layer_id;
    int param_id;
    MLSL::CommReq* req;
    AsyncTask() : layer_id(-1), param_id(-1), req() {};
    AsyncTask(int layer_id, int param_id, MLSL::CommReq* req) :
      layer_id(layer_id), param_id(param_id), req(req) {}
  };

  template <typename Dtype>
  class MultiSync : public MultiSolver<Dtype>::Callback {

    boost::shared_ptr<MultiSolver<Dtype>> solver;

    vector<shared_ptr<Layer<Dtype>>> layers;
    shared_ptr<Net<Dtype>> net;
    const vector<Blob<Dtype> *> &net_params;
    vector<vector<int>> layer_param_ids;
#ifdef FW_OVERLAP_OPT
    vector<vector<bool>> param_ids_finished_flags;
#endif

    // layer_id -> blob_id -> cached blob to restore
    // statistics
    vector<vector<shared_ptr<Blob<Dtype>>>> cached_stats;

    // if use_param_server == true
    vector<MLSL::CommReq*> reduce_req_vec;
    std::list<AsyncTask> reduce_req_list;
    vector<MPI_Request> irecv_req_vec;
    vector<MLSL::CommReq*> broadcast_req_vec;
    vector<bool> irecv_done;
    vector<bool> broadcast_launched;
    std::list<mn::TaskRequest> irecv_req_list;
    boost::shared_ptr<mn::Distribution> distrib_bcast;

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
          if (CAN_USE_PRV_DATA(net_params[layer_param_id])) {
            distrib.bcast<Dtype,MLSL::GT_DATA>(
              net_params[layer_param_id]->mutable_prv_data(),
              net_params[layer_param_id]->prv_data_count());
          } else {
            distrib.bcast<Dtype,MLSL::GT_DATA>(
              net_params[layer_param_id]->mutable_cpu_data(),
              net_params[layer_param_id]->count());
          }
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
            caffe_scal<Dtype>(net_param->count(), 1./distrib.get_data_parts(),
                              net_param->mutable_cpu_data());
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
                   << ", FORWARD OVERLAP OPTIMIZATION IS"
#ifdef FW_OVERLAP_OPT
                   << " ENABLED"
#else
                   << " DISABLED"
#endif
                   << ", WEIGHT GRADIENT COMPRESSION IS"
#ifdef ENABLE_WEIGHT_GRAD_COMPRESSION
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

    void launch_allreduce(int layer_id) {
      boost::shared_ptr<Layer<Dtype>> &layer = layers[layer_id];
      if (layer->layerOp == nullptr) {
        return;
      }

#ifdef FW_OVERLAP_OPT
      std::fill(param_ids_finished_flags[layer_id].begin(),
          param_ids_finished_flags[layer_id].end(),
          false);
#endif

      std::vector<int> &param_ids = layer_param_ids[layer_id];
      for (int i = 0; i < param_ids.size(); ++i) {
        if (!layer->ParamNeedReduce(i)) continue;
        if (CAN_USE_PRV_DIFF(net_params[param_ids[i]])) {
          layer->layerOp->GetParameterSet(i)->StartGradientComm(
              (void *) net_params[param_ids[i]]->mutable_prv_diff());
        } else {
          layer->layerOp->GetParameterSet(i)->StartGradientComm(
              (void *) net_params[param_ids[i]]->mutable_cpu_diff());
        }
      }
    }

    void launch_reduce(int layer_id, int param_id) {
      mn::Distribution& distrib = layers[layer_id]->GetDistribution();
      Dtype* send_buff = NULL;
      Dtype* recv_buff = NULL;
      size_t buf_size = net_params[param_id]->count();
      if (CAN_USE_PRV_DIFF(net_params[param_id])) {
        send_buff = (Dtype*)net_params[param_id]->prv_diff();
        recv_buff = net_params[param_id]->mutable_prv_diff();
      }
      else {
        send_buff = (Dtype*)net_params[param_id]->cpu_diff();
        recv_buff = net_params[param_id]->mutable_cpu_diff();
      }
      reduce_req_vec[param_id] =
        distrib.reduce_async<Dtype,MLSL::ReductionType::RT_SUM,MLSL::GroupType::GT_DATA>(
          send_buff, recv_buff, buf_size);
      if (reduce_req_vec[param_id] != NULL && distrib.is_root(MLSL::GroupType::GT_DATA)) {
        AsyncTask req_task(layer_id, param_id, NULL);
        reduce_req_list.push_back(req_task);
      }
    }

    void check_and_launch_comm_to_ps() {
      std::list<AsyncTask>::iterator iter = reduce_req_list.begin();
      int mpi_rank = mn::get_node_rank();
      while (iter != reduce_req_list.end()) {
        bool complete = false;
        if (reduce_req_vec[iter->param_id] == NULL)
          complete = true;
        else {
          MLSL::Environment::GetEnv().Test(reduce_req_vec[iter->param_id], &complete);
        }
        if (complete) {
          // reset req to indicate no need to do Wait
          reduce_req_vec[iter->param_id] = NULL;

          void* send_buff;
          void* recv_buff;
          int param_id = iter->param_id;
          size_t buf_size = net_params[param_id]->count();
          
          if (CAN_USE_PRV_DIFF(net_params[param_id] ) ) {
            send_buff = (void*)net_params[param_id]->prv_diff();
          }
          else {
            send_buff = (void*)net_params[param_id]->cpu_diff();
          }
          if (CAN_USE_PRV_DATA(net_params[param_id] ) ) {
            recv_buff = (void*)net_params[param_id]->mutable_prv_data();
          }
          else {
            recv_buff = (void*)net_params[param_id]->mutable_cpu_data();
          }
          mn::Distribution &distrib = layers[iter->layer_id]->GetDistribution();
          int server_mpi_rank = mn::param_to_server_rank(iter->layer_id, iter->param_id);
          mn::TaskRequest task(
            mpi_rank, iter->layer_id, iter->param_id,
            distrib.get_node_id(MLSL::GroupType::GT_MODEL),
            distrib.get_nodes_count(MLSL::GroupType::GT_MODEL));
          int tag = task.GetTag();
          MPI_Request send_req;
          int recv_flag = 1;
           // recv from PS
          MPI_Irecv(recv_buff, buf_size, mn::DtypeToMPIDtype<Dtype>(),
                    server_mpi_rank, tag, MPI_COMM_WORLD, &irecv_req_vec[param_id]);
          MPI_Test(&irecv_req_vec[param_id], &recv_flag, MPI_STATUS_IGNORE);
          CHECK(!recv_flag);
          // Send to PS
          MPI_Isend(send_buff, buf_size, mn::DtypeToMPIDtype<Dtype>(),
                    server_mpi_rank, tag, MPI_COMM_WORLD, &send_req);
          // TODO: why do we have to wait here?
          MPI_Wait(&send_req, MPI_STATUS_IGNORE);

          irecv_req_list.push_back(task);
          iter = reduce_req_list.erase(iter);
        }
        else iter++;
      }
    }

    void launch_param_broadcast(int layer_id, int param_id) {
      Dtype* buff;
      if (CAN_USE_PRV_DATA(net_params[param_id])) {
        if (distrib_bcast->is_root(MLSL::GroupType::GT_DATA))
          buff = (Dtype*)net_params[param_id]->prv_data();
        else
          buff = net_params[param_id]->mutable_prv_data();
      }
      else {
        if (distrib_bcast->is_root(MLSL::GroupType::GT_DATA))
          buff = (Dtype*)net_params[param_id]->cpu_data();
        else
          buff = net_params[param_id]->mutable_cpu_data();
      }
      size_t buf_size = net_params[param_id]->count();
      broadcast_req_vec[param_id] =
          distrib_bcast->bcast_async<Dtype,MLSL::GroupType::GT_DATA>(buff, buf_size);
    }

    void check_and_launch_broadcast() {
      std::list<mn::TaskRequest>::iterator iter = irecv_req_list.begin();
      while (iter != irecv_req_list.end() ) {
        int flag = 0;
        int param_id = iter->param_id_;
        if (irecv_done[param_id]) {
          flag = 1;
        } else {
          MPI_Test(&irecv_req_vec[param_id], &flag, MPI_STATUS_IGNORE);
        }
        if (flag) {
          irecv_req_vec[param_id] = MPI_REQUEST_NULL;
          irecv_done[param_id] = true;
          iter = irecv_req_list.erase(iter);
        }
        else
          iter++;
      }
      // Make sure the order of bcast is the same inside the group:
      // Layers and net params in reverse order
      // TODO: relax this ordering constraints for more efficient
      // communication
      for (int i = layers.size() - 1; i >= 0; i--) {
        for (int j = layer_param_ids[i].size() - 1; j >= 0; j--) {
          int param_id = layer_param_ids[i][j];
          if (!broadcast_launched[param_id]) {
            if (irecv_done[param_id]) {
              launch_param_broadcast(i, param_id);
              broadcast_launched[param_id] = true;
            } else return;
          }
        }
      }
    }

    void on_backward_finished(int layer_id) {
      boost::shared_ptr<Layer<Dtype>> &layer = layers[layer_id];
      if (layer->layerOp == nullptr) {
        return;
      }

      if (mn::use_param_server()) {
        std::vector<int> &param_ids = layer_param_ids[layer_id];
        // TODO: descending is faster?
        for (int i = param_ids.size() - 1; i >= 0; --i) {
          if (!layer->ParamNeedReduce(i)) continue;
          launch_reduce(layer_id, param_ids[i]);
          mn::Distribution &distrib = layer->GetDistribution();
          if (distrib.is_root(MLSL::GroupType::GT_DATA)) {
            check_and_launch_comm_to_ps();
            check_and_launch_broadcast();
          } else {
            launch_param_broadcast(layer_id, param_ids[i]);
          }
        }
      } else {
        launch_allreduce(layer_id);
      }
    }

    void delwt_wait_ps(int layer_id) {
      mn::Distribution &distrib = layers[layer_id]->GetDistribution();
      if (distrib.is_root(MLSL::GroupType::GT_DATA)) {
        std::vector<int> &param_ids = layer_param_ids[layer_id];
        // TODO: can we start comm with ps earlier? Per-layer data would be inconsistent then.
        check_and_launch_comm_to_ps();
        check_and_launch_broadcast();
        for (int i = param_ids.size() - 1; i >= 0; i--) {
          int param_id = param_ids[i];
          // wait for reduce
          if (reduce_req_vec[param_id] != NULL) {
            MLSL::Environment::GetEnv().Wait(reduce_req_vec[param_id]);
          }
          reduce_req_vec[param_id] = NULL;
          // wait for new param from param server
          if (irecv_req_vec[param_id] != MPI_REQUEST_NULL) {
            MPI_Wait(&irecv_req_vec[param_id], MPI_STATUS_IGNORE);
            // the req is set to MPI_Request_NULL indicating the request is already finished
            irecv_req_vec[param_id] = MPI_REQUEST_NULL;
          }
          irecv_done[param_id] = false;
          // wait for the completion of broadcast
          if (broadcast_req_vec[param_id] != NULL) {
            MLSL::Environment::GetEnv().Wait(broadcast_req_vec[param_id]);
            broadcast_req_vec[param_id] = NULL;
          }
          broadcast_launched[param_id] = false;
        }
      }
#ifdef FW_OVERLAP_OPT
      solver->set_layer_finished_flag(layer_id, true);
#endif
    }

    void delwt_wait_no_ps(int layer_id) {
      boost::shared_ptr<Layer<Dtype>> &layer = layers[layer_id];
      if (layer->layerOp == nullptr) {
#ifdef FW_OVERLAP_OPT
        solver->set_layer_finished_flag(layer_id, true);
#endif
        return;
      }

      std::vector<int> &param_ids = layer_param_ids[layer_id];
      for (int i=0; i<param_ids.size(); i++) {
        if (!layer->ParamNeedReduce(i)
#ifdef FW_OVERLAP_OPT
            || (param_ids_finished_flags[layer_id][i] == true)) {
          param_ids_finished_flags[layer_id][i] = true;
#else
          ) {
#endif
          continue;
        }

#ifdef FW_OVERLAP_OPT
        bool is_completed = false;
        Dtype *delwt_buf{(Dtype *) layer->layerOp->GetParameterSet(i)->TestGradientComm(&is_completed)};
#else
        Dtype *delwt_buf{(Dtype *) layer->layerOp->GetParameterSet(i)->WaitGradientComm()};
#endif
        if (delwt_buf) {
#ifdef FW_OVERLAP_OPT
          assert(is_completed);
          param_ids_finished_flags[layer_id][i] = true;
#endif
          if (CAN_USE_PRV_DIFF(net_params[param_ids[i]])) {
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

#ifdef FW_OVERLAP_OPT
      int finished_count = std::count(param_ids_finished_flags[layer_id].begin(),
            param_ids_finished_flags[layer_id].end(), true);
      if (finished_count == param_ids.size()) {
        solver->set_layer_finished_flag(layer_id, true);
      }
#endif
    }

    void on_delwt_wait(int layer_id) {
      if (mn::use_param_server()) {
        delwt_wait_ps(layer_id);
      } else {
        delwt_wait_no_ps(layer_id);
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
