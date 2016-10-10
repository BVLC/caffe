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

#ifdef USE_MPI
#include <mpi.h>
#include "caffe/internode/mpiutil.hpp"
#endif
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>
#include <string>
#include <utility>
#include <vector>
#include "caffe/internode/broadcast_callback.hpp"
#include "caffe/internode/mpi_configuration.hpp"
#include "caffe/multinode/multinode.hpp"
#include "caffe/serialization/BlobCodec.hpp"

#ifdef USE_MPI

const int MSG_TAG = 1972;

struct MpiNode {
  enum Type {
    PARAM_SERVER, DATA_SERVER, CLIENT
  };
};

int mpi_param_server_proc_rank;
#endif

namespace caffe {
namespace internode {

#ifdef USE_MPI
extern boost::asio::io_service& get_io_service(boost::shared_ptr<Daemon>);

typedef boost::function<void(bool, int, int)> RequestCallback;
typedef std::pair<MPI_Request, RequestCallback> MpiRequestWithCallback;

void mpi_register_node_rank(MpiNode::Type node_type) {
  MPI_Status status;
  MPI_Comm new_comm;
  MPI_Group new_group;
  int buf = 0, rank = 0, size = 0, tag = 10;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Comm_group(MPI_COMM_WORLD, &new_group);
  MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

  if (node_type == MpiNode::PARAM_SERVER) {
    for (int i = 0; i < size; i++) {
      if (i == rank)
        mpi_param_server_proc_rank = rank;
      else
        MPI_Send(&rank, 1, MPI_INT, i, tag, new_comm);
    }
  } else {
    MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, tag, new_comm, &status);
    mpi_param_server_proc_rank = buf;
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

namespace {

class MpiClient : public Waypoint {
  std::vector<Handler*> handlers;
  std::vector<MpiRequestWithCallback> requests;
  std::vector<char> buffer;
  boost::recursive_mutex mtx;

  void set_recv() {
    requests.push_back(std::make_pair(
      MPI_Request(),
      boost::bind(&MpiClient::received, this, _1, _2, _3)));
    MPI_Irecv(
            &buffer.front(),
            buffer.size(),
            MPI_CHAR,
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            MPI_COMM_WORLD,
            &requests.back().first);
  }

  void received(bool ok, int size, int sender) {
    if (ok) {
      for (int i = 0; i < handlers.size(); ++i) {
        handlers[i]->received(&buffer.front(), size, this);
      }
    }

    set_recv();
  }

 public:
  MpiClient(const std::string& server_name, size_t max_buffer_size)
  : buffer(max_buffer_size) {
    set_recv();
  }

  // Called from external, data computed by layer
  virtual void async_send(const char* buffer,
          size_t size,
          SentCallback callback) {
    boost::recursive_mutex::scoped_lock lock(mtx);

    requests.push_back(std::make_pair(
        MPI_Request(), boost::bind(callback, _1)));
      MPI_Isend(
        const_cast<char*>(buffer),
        size,
        MPI_CHAR,
        mpi_param_server_proc_rank,
        MSG_TAG,
        MPI_COMM_WORLD,
        &requests.back().first);
  }

  virtual void register_receive_handler(Handler* handler) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    handlers.push_back(handler);
  }

  virtual RemoteId id() const {
    return 0;
  }

  virtual string address() const {
    return "";
  }

  virtual bool guaranteed_comm() const {
    return true;
  }

  virtual size_t max_packet_size() const {
    return buffer.size();
  }

  virtual void poll_one(shared_ptr<Daemon> daemon) {
    for (int i = 0; i < requests.size(); ++i) {
      MPI_Status status;
      int flag = 0, result = MPI_Test(&requests[i].first, &flag, &status);
      if (result == MPI_SUCCESS && flag) {
        int size = 0, sender = status.MPI_SOURCE;
        result = MPI_Get_count(&status, MPI_CHAR, &size);
        if (result == MPI_SUCCESS) {
          MpiRequestWithCallback request = requests[i];
          requests.erase(requests.begin() + i);
          request.second(true, size, sender);
          break;
        } else {
          LOG(ERROR) << mpi_get_error_string(result);
        }
      }
    }

    post(daemon);
  }

  virtual void post(shared_ptr<Daemon> daemon) {
    get_io_service(daemon).post(
      boost::bind(&MpiClient::poll_one, this, daemon));
  }
};


class MpiClientFromServer : public Waypoint {
  const size_t buffer_size;
  std::vector<Waypoint::Handler*> receive_handlers;
  boost::shared_ptr<std::vector<MpiRequestWithCallback> > requests;
  boost::recursive_mutex mtx;
  int rank;

 public:
  MpiClientFromServer(
    boost::shared_ptr<Daemon> daemon,
    boost::shared_ptr<std::vector<MpiRequestWithCallback> > requests,
    int rank,
    size_t buffer_size)
  : buffer_size(buffer_size)
  , requests(requests)
  , rank(rank) {
  }

  // Send data received from server to client (?)
  virtual void async_send(const char* buffer,
          size_t size,
          SentCallback callback) {
    boost::recursive_mutex::scoped_lock lock(mtx);

    requests->push_back(std::make_pair(
      MPI_Request(), boost::bind(callback, _1)));
    MPI_Isend(
      const_cast<char*>(buffer),
      size,
      MPI_CHAR,
      rank,
      MSG_TAG,
      MPI_COMM_WORLD,
      &requests->back().first);
  }

  virtual void register_receive_handler(Handler* handler) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    receive_handlers.push_back(handler);
  }

  virtual RemoteId id() const {
    return (size_t) rank;
  }

  virtual string address() const {
    return boost::lexical_cast<string>(rank);
  }

  virtual bool guaranteed_comm() const {
    return true;
  }

  virtual size_t max_packet_size() const {
    return buffer_size;
  }
};


class MpiServer : public MultiWaypoint {
  boost::shared_ptr<Daemon> daemon;
  std::vector<Handler*> accept_handlers;
  std::vector<Waypoint::Handler*> receive_handlers;
  std::vector<MpiRequestWithCallback> requests;
  std::vector<char> buffer;
  std::vector< boost::shared_ptr<MpiClientFromServer> > clients_from_server;
  boost::recursive_mutex mtx;

  void set_recv() {
    boost::recursive_mutex::scoped_lock lock(mtx);

    requests.push_back(
      std::make_pair(
        MPI_Request(),
        boost::bind(&MpiServer::received, this, _1, _2, _3)));
    MPI_Irecv(&buffer.front(),
              buffer.size(),
              MPI_CHAR,
              MPI_ANY_SOURCE,
              MPI_ANY_TAG,
              MPI_COMM_WORLD,
              &requests.back().first);
  }

  void received(bool ok, int size, int sender) {
    boost::recursive_mutex::scoped_lock lock(mtx);

    if (ok) {
      for (int i = 0; i < receive_handlers.size(); ++i) {
        receive_handlers[i]->received(
          &buffer.front(), size, clients_from_server.at(sender - 1).get());
      }
    }

    set_recv();
  }

 public:
  MpiServer(boost::shared_ptr<Daemon> daemon,
            const std::string& server_name,
            size_t max_buffer_size)
  : daemon(daemon)
  , buffer(max_buffer_size) {
    int num_ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Create clients
    for (int i = 0; i < num_ranks; ++i) {
      if (i == mpi_param_server_proc_rank) continue;
      clients_from_server.push_back(
          boost::make_shared<MpiClientFromServer>(
            daemon,
            boost::make_shared< std::vector<MpiRequestWithCallback> >(requests),
            i,
            buffer.size()) );
    }

    set_recv();
  }

  virtual void async_send(const char* buffer,
          size_t size,
          SentCallback callback) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    BroadcastCallback<SentCallback> broadcast_callback(callback);
    for (int i = 0; i < clients_from_server.size(); ++i) {
      requests.push_back(std::make_pair(MPI_Request(), broadcast_callback));
      MPI_Isend(
        const_cast<char*>(buffer),
        size,
        MPI_CHAR,
        clients_from_server.at(i)->id(),
        MSG_TAG,
        MPI_COMM_WORLD,
        &requests.back().first);
    }
  }

  // Called from external SynchronousParamServer > configuration.cpp:poll_one()
  void accept_all() {
    for (int i = 0; i < clients_from_server.size(); ++i)
      accept_handlers.back()->accepted(clients_from_server[i]);
  }

  // Called from external code, handler is called for client connection
  virtual void register_peer_change_handler(Handler* handler) {
    boost::recursive_mutex::scoped_lock lock(mtx);

    accept_handlers.push_back(handler);

    get_io_service(daemon).post(boost::bind(&MpiServer::accept_all, this));
  }

  // For BLOB data receiving from external layer
  virtual void register_receive_handler(Waypoint::Handler* handler) {
    boost::recursive_mutex::scoped_lock lock(mtx);

    receive_handlers.push_back(handler);

    for (int i = 0; i < clients_from_server.size(); ++i) {
      clients_from_server.at(i)->register_receive_handler(handler);
    }
  }

  virtual RemoteId id() const {
    return mpi_param_server_proc_rank;
  }

  virtual string address() const {
    return boost::lexical_cast<std::string>(mpi_param_server_proc_rank);
  }

  virtual bool guaranteed_comm() const {
    return true;
  }

  virtual size_t max_packet_size() const {
    return buffer.size();
  }

  // Get data from param server
  // ( SynchronousParamServer.run() while.poll_one() ) and send to client
  virtual void poll_one(shared_ptr<Daemon> daemon) {
    for (int i = 0; i < requests.size(); ++i) {
      MPI_Status status;
      int flag = 0;
      int result = MPI_Test(&requests[i].first, &flag, &status);
      if (result == MPI_SUCCESS && flag) {
        int size = 0, sender = status.MPI_SOURCE;
        result = MPI_Get_count(&status, MPI_CHAR, &size);
        if (result == MPI_SUCCESS) {
          MpiRequestWithCallback request = requests[i];
          requests.erase(requests.begin() + i);
          request.second(result == MPI_SUCCESS, size, sender);
          break;
        } else {
          LOG(ERROR) << mpi_get_error_string(result);
        }
      }
    }

    post(daemon);
  }

  virtual void post(shared_ptr<Daemon> daemon) {
    get_io_service(daemon).post(
      boost::bind(&MpiServer::poll_one, this, daemon));
  }
};

}  // namespace
#endif

// Called from 'caffe train' command
boost::shared_ptr<Waypoint> configure_client(
        boost::shared_ptr<Daemon> communication_daemon,
        std::string server_name,
        size_t max_buffer_size) {
#ifdef USE_MPI
  mpi_register_node_rank(MpiNode::CLIENT);

  boost::shared_ptr<MpiClient> client(
    new MpiClient(server_name, max_buffer_size));
  client->post(communication_daemon);

  return client;
#else
  throw std::runtime_error("Can't configure MPI with not definied USE_MPI");
#endif
}

boost::shared_ptr<MultiWaypoint> configure_server(
        boost::shared_ptr<Daemon> communication_daemon,
        std::string server_name,
        size_t max_buffer_size) {
#ifdef USE_MPI
  mpi_register_node_rank(MpiNode::PARAM_SERVER);

  boost::shared_ptr<MpiServer> server(
    new MpiServer(communication_daemon, server_name, max_buffer_size));
  server->post(communication_daemon);

  return server;
#else
  throw std::runtime_error("Can't configure MPI with not definied USE_MPI");
#endif
}

}  // namespace internode
}  // namespace caffe
