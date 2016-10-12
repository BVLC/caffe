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

#ifndef CAFFE_INTERNODE_TREE_CLUSTER_HPP_
#define CAFFE_INTERNODE_TREE_CLUSTER_HPP_

#include <boost/function.hpp>
#include <string>
#include <vector>

namespace caffe {
namespace internode {

class Daemon;

typedef size_t RemoteId;

class TreeWaypoint {
 public:
  struct Handler {
    virtual void received_from_parent(
            char* buffer, size_t size) = 0;
    virtual void received_from_child(
            char* buffer, size_t size, RemoteId id) = 0;
  };

  static TreeWaypoint* get_instance();

  virtual boost::shared_ptr<Daemon> get_daemon() = 0;
  virtual void set_buffer_size(size_t max_packet_size) = 0;

  typedef boost::function<void(bool succesful) > SentCallback;
  virtual void async_send_to_parent(
          const char* buffer, size_t size, SentCallback) = 0;
  virtual void async_send_to_children(
          const char* buffer, size_t size, SentCallback) = 0;

  virtual void register_receive_handler(Handler* handler) = 0;

  virtual RemoteId id() const = 0;
  virtual std::vector<RemoteId> children() const = 0;
  virtual RemoteId parent() const = 0;
  virtual int total_nodes() const = 0;
  virtual void lets_die_together() const = 0;
  virtual bool is_finished() const = 0;
};
}  // namespace internode
}  // namespace caffe

#endif  // CAFFE_INTERNODE_TREE_CLUSTER_HPP_
