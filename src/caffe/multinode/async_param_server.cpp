/*
All modification made by Intel Corporation: © 2017 Intel Corporation

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

// Initial implementation from Jian Zhang and Ioannis Mitliagkas, Stanford, on Oct 2 2016
// Refer to the paper: https://arxiv.org/pdf/1708.05256.pdf

#ifdef USE_MLSL

#include <cstdlib>
#include <climits>
#include <boost/make_shared.hpp>
#include <thread>

#include "caffe/caffe.hpp"
#include "caffe/multinode/async_param_server.hpp"

namespace caffe {
  namespace mn {

    using std::make_pair;

    template <typename Dtype>
    AsyncParamServer<Dtype>::AsyncParamServer(boost::shared_ptr<Solver<Dtype> > solver) :
      recv_tasks_iter_(0), 
      solver_(solver),
      send_cnt_(0), update_cnt_(0) {

      // setup the mpi buffers and recv task vector
      int mpi_rank = get_node_rank();
      shared_ptr<Net<Dtype>> net = solver_->net();
      const vector<Blob<Dtype> *> &net_params = net->learnable_params();
      
      for (int i = 0; i < get_num_groups(); i++) {
        int root_rank = get_group_root_rank(i);
        //iterate over layers and skip the ones without params
        for (int j = 0; j < net->layers().size(); j++) {
          shared_ptr<Layer<Dtype>> layer = net->layers()[j];
          //skip layers w/o parameters
          if ((layer->layerOp == nullptr) || !(layer->layerOp->HasParameterSets())) {
            continue;
          }
          const MultinodeLayerParameter & mn_layer_param = layer->layer_param().multinode();
          int model_parts = mn_layer_param.model_parts();
          int mn_num_nodes = mn_layer_param.num_nodes();
          GetCanonicalMnParam(mn_num_nodes, model_parts);
          vector<int> layer_param_ids = net->get_layer_learnable_param_ids(j);
          for (int k = 0; k < layer_param_ids.size(); k++) {
            int param_id = layer_param_ids[k];
            if (!layer->ParamNeedReduce(k)) continue;
            if (param_to_server_rank(j, param_id) != mpi_rank) continue;
            Blob<Dtype> *blob = net_params[param_id];
            // Setup buf for recv
            Dtype* buf = (Dtype*)std::malloc(sizeof(Dtype) * blob->count());
            recv_buf_[make_pair(root_rank, param_id)] = make_pair(buf, blob->count());
            for (int part_id = 0; part_id < model_parts; part_id++) {
              int part_root_rank = get_group_root_rank(i, part_id, model_parts);
              int64_t part_offset = part_id * blob->count() / model_parts;
              TaskRequest recv_task(part_root_rank, j, param_id, part_id, model_parts);
              recv_tasks_.push_back(recv_task);
              rank_layer_blob_to_vec_pos[make_pair(part_root_rank, param_id)] =
                recv_tasks_.size() - 1;
              MPI_Irecv(buf + part_offset, blob->count() / model_parts,
                        DtypeToMPIDtype<Dtype>(), part_root_rank,
                        recv_task.GetTag(), MPI_COMM_WORLD,
                        &(recv_tasks_[recv_tasks_.size() - 1].mpi_request_));
              async_iter_[make_pair(param_id, part_id)] = solver_->iter();
            }
            // Setup buf for send
            buf = (Dtype*)std::malloc(sizeof(Dtype) * blob->count());
            send_buf_[make_pair(root_rank, param_id)] = make_pair(buf, blob->count());
          }
        }
      }
      total_update_ = total_send_ = recv_tasks_.size() * (solver_->param().max_iter() - 1);
    }

    template <typename Dtype>
    AsyncParamServer<Dtype>::~AsyncParamServer() {
      // clean mpi buffers
      shared_ptr<Net<Dtype>> net = solver_->net();
      for (int i = 0; i < get_num_groups(); i++) {
        int root_rank = get_group_root_rank(i);
        for (int j = 0; j < net->layers().size(); j++) {
          vector<int> layer_param_ids = net->get_layer_learnable_param_ids(j);
          for (int k = 0; k < layer_param_ids.size(); k++) {
            pair<int,int> key = make_pair(root_rank, layer_param_ids[k]);
            if (send_buf_.find(key) != send_buf_.end()) {
              std::free(send_buf_[key].first);
            }
            if (recv_buf_.find(key) != recv_buf_.end()) {
              std::free(recv_buf_[key].first);
            }
          }
        }
      }
    }

    // TODO Jian how to get the correct iter number potentially get the version and set iter before update
    template <typename Dtype>
    void AsyncParamServer<Dtype>::ProcessUpdateTask() {
      const vector<Blob<Dtype> *> &net_params = solver_->net()->learnable_params();
      std::deque<TaskRequest> to_update;
      update_queue_mutex_.lock();
      to_update.swap(update_tasks_);
      update_queue_mutex_.unlock();
      while (!to_update.empty() ) {
        TaskRequest task = to_update.front();
        to_update.pop_front();

        // copy to diff in solver
        int root_rank = world_rank_to_root_rank(task.part_root_rank_);
        Blob<Dtype>* blob = net_params[task.param_id_];
        Dtype* solver_diff = blob->mutable_cpu_diff();
        Dtype* mpi_buf = 
          recv_buf_[make_pair(root_rank, task.param_id_)].first;
        int64_t count = 
          recv_buf_[make_pair(root_rank, task.param_id_)].second;
        CHECK(count == blob->count() );
        //copy MPI buffer to solver_diff
        int64_t part_offset = task.part_id_ * count / task.num_parts_;
        caffe_copy(count / task.num_parts_,
                   mpi_buf + part_offset, solver_diff + part_offset);
        // apply update
        int blob_wise_iter = async_iter_[make_pair(task.param_id_, task.part_id_) ];
        solver_->set_iter(blob_wise_iter);
        // TODO: supports partial param update per model parts
        solver_->ApplyUpdate(task.param_id_);

        //clean up
        solver_->net()->ClearParamDiffs(task.param_id_);
        async_iter_[ make_pair(task.param_id_, task.part_id_) ] += 1;
        update_cnt_ += 1;
        
        // copy model(data) in solver to mpi buffer
        mpi_buf = send_buf_[make_pair(root_rank, task.param_id_)].first;
        caffe_copy(count / task.num_parts_,
                   blob->cpu_data() + part_offset, mpi_buf + part_offset);

        //ship off
        send_queue_mutex_.lock();
        send_tasks_.push_back(task);
        send_queue_mutex_.unlock();
      }
    }


    template <typename Dtype>
    void AsyncParamServer<Dtype>::ProcessSendTask() {
      std::deque<TaskRequest> to_send;
      send_queue_mutex_.lock();
      to_send.swap(send_tasks_);
      send_queue_mutex_.unlock();
      std::vector<MPI_Request> send_request;
      while (!to_send.empty() ) {
        TaskRequest task = to_send.front();
        to_send.pop_front();

        int root_rank = world_rank_to_root_rank(task.part_root_rank_);
        int param_id = task.param_id_;
        int part_id = task.part_id_;
        int tag = task.GetTag();

        // start a new listening to wait for message from roots
        Dtype* recv_ptr = recv_buf_[make_pair(root_rank, param_id)].first;
        int count = recv_buf_[make_pair(root_rank, param_id)].second;
        int64_t part_offset = part_id * count / task.num_parts_;
        int vec_pos = rank_layer_blob_to_vec_pos[make_pair(task.part_root_rank_, param_id)];
        MPI_Irecv(recv_ptr + part_offset, count / task.num_parts_, DtypeToMPIDtype<Dtype>(),
                  task.part_root_rank_, tag, MPI_COMM_WORLD, &(recv_tasks_[vec_pos].mpi_request_) );
        
#ifdef DEBUG_ORDER_BCAST
        DEBUG_INFO("PS send message for layer ") << layer_id << " to rank " << root_rank << " for tag " << tag << std::endl;
#endif

        //prepare the matching send now
        std::pair<Dtype*, int64_t> buf = send_buf_[make_pair(root_rank, param_id)];
        Dtype* send_ptr = buf.first;
        // We do not need to care about the request. Because if the blocking recv
        // has not finished on root, it will not start a new send task
        // MPI_Request send_request;
        send_request.push_back(MPI_Request() );
        MPI_Isend(send_ptr + part_offset, count / task.num_parts_, DtypeToMPIDtype<Dtype>(),
                  task.part_root_rank_, tag, MPI_COMM_WORLD, &(send_request.back() ) );

        //increase sent count
        send_cnt_ += 1;
      }
      if (send_request.size() != 0) {
        MPI_Waitall(send_request.size(), &send_request[0], MPI_STATUSES_IGNORE);
      }
    }


    template <typename Dtype>
    void AsyncParamServer<Dtype>::ProcessRecvTask() {
      int flag = 0;
      for (int i = 0; i < recv_tasks_.size(); i++) {
        if (recv_tasks_[recv_tasks_iter_].mpi_request_ != MPI_REQUEST_NULL) {
          MPI_Test(&(recv_tasks_[recv_tasks_iter_].mpi_request_), &flag, MPI_STATUS_IGNORE);
          if (flag) {
            // currently no need to lock the solver buffer, as comp thread
            // takes care of two copy operations.
            update_queue_mutex_.lock();
            update_tasks_.push_back(recv_tasks_[recv_tasks_iter_] );
            update_queue_mutex_.unlock();
          }
        }
        recv_tasks_iter_ = (recv_tasks_iter_ + 1) % recv_tasks_.size();
        if (flag) return;
      }
    }


    template <typename Dtype>
    void AsyncParamServer<Dtype>::ComputeLoop() {
      do {
        ProcessUpdateTask();
      } while(update_cnt_ < total_update_);
    }


    template <typename Dtype>
    void AsyncParamServer<Dtype>::CommLoop() {
      do {
        ProcessSendTask();
        ProcessRecvTask();
      } while(send_cnt_ < total_send_);
    }

    template <typename Dtype>
    void AsyncParamServer<Dtype>::Run() {
      // spawn compute thread
      std::thread compute_thread(&AsyncParamServer<Dtype>::ComputeLoop, this);
      // spawn communication thread
      CommLoop();
      compute_thread.join();
    }

    INSTANTIATE_CLASS(AsyncParamServer);
  } // end of namespace mn
 
} // end of namespace caffe
#endif
