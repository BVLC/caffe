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

#ifndef ASYNC_PARAM_SERVER_H
#define ASYNC_PARAM_SERVER_H

#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <boost/thread.hpp>
#include <cstdlib>
#include <mpi.h>

#include "caffe/caffe.hpp"
#include "caffe/multinode/multi_solver.hpp"

namespace caffe {
  namespace mn {

    using std::make_pair;

    // TODO modify decoding strategy
    // we use TAG = param_id * 10 + part_id + 1973 to identify parts location
    struct TaskRequest {
      int part_root_rank_;
      int layer_id_; 
      int param_id_;
      int part_id_;
      int num_parts_;
      MPI_Request mpi_request_;
  
      TaskRequest(): part_root_rank_(0), layer_id_(0), param_id_(0), part_id_(0),
                     num_parts_(1), mpi_request_() {}
      TaskRequest(int root_rank, int layer_id, int param_id, int part_id, int num_parts) :
        part_root_rank_(root_rank), layer_id_(layer_id), param_id_(param_id), part_id_(part_id),
        num_parts_(num_parts) {}

      int GetTag() {
        return param_id_ * 10 + part_id_ + 1973;
      }
    };

    // protocol:
    // when get a non-blocking mpi receive, comm thread submit a job to the 
    // update_tasks_ queue. 
    // The compute thread will check the update_tasks_ queue. After it finishes
    // update, the compute thread will submit request to send_tasks_ queue.
    // In the communicate loop, the thead consider send task first, and then 
    // process receive tasks.
    template <typename Dtype>
    class AsyncParamServer {
    public:
      AsyncParamServer(boost::shared_ptr<Solver<Dtype> > solver);
      ~AsyncParamServer();
      // in the update task, the compute thread 
      // 0. lock the mutex on blob
      // 1. copy buffer to solvers diff buffer
      // 2. perform updates
      // 3. copy the model to the corresponding mpi buffer
      // 4. submit a send task
      // 5. unlock the mutex blob
      void ProcessUpdateTask();
      // in the Send task, we use non-blocking send for model parts going back to roots
      // We do not need to care about the request. Because if the blocking recv
      // has not finished on root, it will not start a new send task
      void ProcessSendTask();
      // We iterate over the recv_tasks_ vector, when the request is done, we start a
      // new corresponding MPI non-blocking recv call.
      void ProcessRecvTask();
      void ComputeLoop();
      void CommLoop();
      void Run();

    private:
      // for communication
      std::deque<TaskRequest> update_tasks_;
      std::deque<TaskRequest> send_tasks_;
      boost::mutex update_queue_mutex_;
      boost::mutex send_queue_mutex_;
      int recv_tasks_iter_;
      std::vector<TaskRequest> recv_tasks_;
      // part_root_rank, param_id
      std::map<std::pair<int, int>, int> rank_layer_blob_to_vec_pos;
      // root_rank, param_id
      std::map<std::pair<int, int>, std::pair<Dtype*, int64_t> > recv_buf_;
      std::map<std::pair<int, int>, std::pair<Dtype*, int64_t> > send_buf_;

      // for computation
      boost::shared_ptr<Solver<Dtype> > solver_;

      // for termination: count the number of operations 
      // needed in total
      int64_t send_cnt_;
      int64_t update_cnt_; 
      int64_t total_send_;
      int64_t total_update_;

      // iter for different blobs
      // param_id, part_id
      std::map<std::pair<int, int>, int64_t> async_iter_;

    };

  } // end of namespace async_param_server

} // end of namespace caffe

#endif /*ASYNC_PARAM_SERVER_H*/
