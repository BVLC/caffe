

#ifndef MULTI_NODE_NODE_ENV_H_
#define MULTI_NODE_NODE_ENV_H_

#include <string>

#include "caffe/multi_node/msg.hpp"
#include "caffe/caffe.hpp"
#include "caffe/multi_node/sk_sock.hpp"

#include "boost/lexical_cast.hpp"

#include "boost/unordered_map.hpp"
using boost::unordered_map;

#include "boost/thread/once.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/thread.hpp"

#include <sys/types.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <net/if.h>


namespace caffe {

class NodeEnv {

////NOTE: all the public funcations are constants

public:
  const string& IP() { return node_ip_; }
  const string& Interface() { return interface_; }
  
  //ID is used by ZMQ to identify socket
  int ID() { return node_id_; }

  ///
  const SolverParameter& SolverParam() { return solver_param_; }


  static NodeEnv *Instance();

  ////
  const vector<string>& SubAddrs() { return sub_addrs_; }
  const vector<string>& DealerAddrs() { return dealer_addrs_; }
  const vector<string>& ClientAddrs() { return client_addrs_; }
  const vector<int>& DealerIDs() { return dealer_ids_; }
  const vector<string>& BottomAddrs() { return bottom_addrs_; }
  const vector<int>& BottomIDs() { return bottom_ids_; }

  int NumInputBlobs() { return num_input_blobs_; }

  string PubAddr() {
    string addr = "tcp://*:";
    addr += boost::lexical_cast<string>(pub_port_);
    
    return addr;
  }


  string RouterAddr() {
    string addr = "tcp://*:";
    addr += boost::lexical_cast<string>(router_port_);

    return addr;
  }


  void *FindSolver(int64_t msg_id) {
    boost::mutex::scoped_lock lock(id_map_mutex_);
    unordered_map<int64_t, void *>::iterator iter = id_to_solver_.find(msg_id);

    if (iter == id_to_solver_.end()) {
      return NULL;
    } else {
      return iter->second;
    }
  }


  void PutSolver(int64_t msg_id, void *solver) {
    boost::mutex::scoped_lock lock(id_map_mutex_);
    unordered_map<int64_t, void *>::iterator iter = id_to_solver_.find(msg_id);

    CHECK (iter == id_to_solver_.end());
    id_to_solver_[msg_id] = solver;

    return;
  }


  void *DeleteSolver(int64_t msg_id) {
    boost::mutex::scoped_lock lock(id_map_mutex_);
    unordered_map<int64_t, void *>::iterator iter = id_to_solver_.find(msg_id);

    CHECK (iter != id_to_solver_.end());
    
    void *p = iter->second;
    id_to_solver_.erase(iter);

    return p;
  }


  void PushFreeSolver(void *solver) {
    boost::mutex::scoped_lock lock(id_map_mutex_);
    free_solver_.push_back(solver);
  }


  void *PopFreeSolver() {
    boost::mutex::scoped_lock lock(id_map_mutex_);
    
    //the 0th solver is root solver
    if (free_solver_.size() <= 1) {
      return NULL;
    }

    void *p = free_solver_.back();
    free_solver_.pop_back();

    return p;
  }


  int NumFreeSolver() {
    boost::mutex::scoped_lock lock(id_map_mutex_);
    return free_solver_.size();
  }
  
  void *GetRootSolver() {
    if (free_solver_.size() <= 0) {
      return NULL;
    } else {
      return free_solver_[0];
    }
  }
  
  int GetTrainBatchSize() {
    return rt_info_.train_batch_size();
  }
  

public:
  static inline void set_id_server(const string& addr) {
    id_server_addr_ = addr;
  }

  static inline void set_model_server(const string& addr) {
    model_server_addr_ = addr;
  }

  static inline void set_request_file(const string& addr) {
    request_file_addr_ = addr;
  }

  static inline void set_node_role(const NodeRole role) {
    node_role_ = role;
  }

  static inline const string& id_server_addr() {
    return id_server_addr_;
  }

  static inline const string& model_server_addr() {
    return model_server_addr_;
  }

  static inline const string& request_file_addr() {
    return request_file_addr_;
  }

  static inline NodeRole node_role() {
    return node_role_;
  }

protected:
  static std::string IP(const std::string& interface) {
    struct ifaddrs * ifAddrStruct = NULL;
    struct ifaddrs * ifa = NULL;
    void * tmpAddrPtr = NULL;
    std::string ret_ip;

    getifaddrs(&ifAddrStruct);
    for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == NULL) continue;
      if (ifa->ifa_addr->sa_family==AF_INET) {
        // is a valid IP4 Address
        tmpAddrPtr=&((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
        char addressBuffer[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
        if (strncmp(ifa->ifa_name,
                    interface.c_str(),
                    interface.size()) == 0) {
          ret_ip = addressBuffer;
          break;
        }
      }
    }
    if (ifAddrStruct != NULL) freeifaddrs(ifAddrStruct);
    return ret_ip;
  }

  static void GetInterfaceAndIP(std::string& interface, std::string& ip) {
    struct ifaddrs * ifAddrStruct = NULL;
    struct ifaddrs * ifa = NULL;

    interface.clear();
    ip.clear();
    getifaddrs(&ifAddrStruct);
    for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
      if (NULL == ifa->ifa_addr) continue;

      if (AF_INET == ifa->ifa_addr->sa_family &&
        0 == (ifa->ifa_flags & IFF_LOOPBACK)) {

        char address_buffer[INET_ADDRSTRLEN];
        void* sin_addr_ptr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
        inet_ntop(AF_INET, sin_addr_ptr, address_buffer, INET_ADDRSTRLEN);

        ip = address_buffer;
        interface = ifa->ifa_name;

        break;
      }
    }

    if (NULL != ifAddrStruct) freeifaddrs(ifAddrStruct);
    return;
  }


protected: 
  static void InitNode(void);
  int InitNodeID();
  int InitModel();
  int InitIP();


private:
  NodeEnv() {
    node_id_ = INVALID_ID;
  }


protected:
  int node_id_;
  string node_ip_;
  string interface_;

  int router_port_;
  int pub_port_;

  int server_port_;
  
  //parameters from server
  SolverParameter solver_param_;
  RouteInfo rt_info_;

  //routing addresses

  //addresses of the SUB sockets to receive broadcasting messages
  vector<string> sub_addrs_;

  //address of the dealer sockets to send message to upstream node
  vector<string> dealer_addrs_;
  
  //ID of the dealers
  vector<int> dealer_ids_;

  //address of the clients
  vector<string> client_addrs_;

  //listen address in the loss layer
  vector<string> bottom_addrs_;
  
  //node ID in the bottom nodes, used for routing
  vector<int> bottom_ids_;

  //number of input blobs
  int num_input_blobs_;
  
  //mapping message id to solver
  unordered_map<int64_t, void *> id_to_solver_;
  boost::mutex  id_map_mutex_;
  
  //available solvers
  vector<void *> free_solver_;

private:
  //For Singleton pattern
  static NodeEnv *instance_;
  static boost::once_flag once_;

  // get unique integer id from id server
  static string id_server_addr_;

  static string model_server_addr_;
  
  // request file is sent to model server for model splitting
  static string request_file_addr_;

  static NodeRole node_role_;

DISABLE_COPY_AND_ASSIGN(NodeEnv);
};

} //namespace caffe

#endif


