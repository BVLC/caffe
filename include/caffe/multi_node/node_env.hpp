

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

const int TRAIN_NOTIFY_INTERVAL = 100;

class NodeEnv {

////NOTE: all the public funcations are constants
public:
  static NodeEnv *Instance();

public:
  const string& IP() { return node_ip_; }
  const string& Interface() { return interface_; }
  
  //ID is used by ZMQ to identify socket
  int ID() { return node_id_; }

  ///
  const SolverParameter& SolverParam() { return solver_param_; }
  
  SolverParameter* mutable_SolverParam() { return &solver_param_; }

  /// For connection with upstream nodes
  const vector<string>& sub_addrs() { return sub_addrs_; }
  const vector<string>& prev_router_addrs() { return prev_router_addrs_; }
  const vector<int>& prev_node_ids() { return prev_node_ids_; }
  
  int batch_size() {
    const NetParameter& net_param = solver_param_.net_param();
    if (net_param.input_shape_size() <= 0) {
      return -1;
    } else {
      // TODO: add a formal dim here
      return net_param.input_shape(0).dim(0);
    }
  }
  
  bool is_fc_gateway() {
    return node_info_.node_role() == FC_GATEWAY;
  }
  
  int node_position() { return node_info_.position(); }

  /// We broadcast blobs to downstream nodes
  const vector<string>& bcast_addrs() { return bcast_addrs_; }
  
  /// for forwarding some blobs
  const vector<string>& forward_addrs() { return fwrd_addrs_; }
  const vector<int>& forward_ids() { return fwrd_ids_; }
   /// name of blobs that need to be forwarded
  const vector<vector<string> >& forward_blobs() { return fwrd_blobs_; }
 
  const vector<string>& gateway_addrs() { return gateway_addrs_; }
  const vector<int>& gateway_ids() { return gateway_ids_; }
  const vector<vector<string> >& gateway_blobs() { return gateway_blobs_; }
  
  // for parameter server nodes
  const vector<string>& ps_addrs() { return ps_addrs_; }
  const vector<int>& ps_ids() { return ps_ids_; }

  const vector<string>& FindPSLayer(int ps_id) {
    map<int, int>::iterator iter = ps_id_to_layers_id_.find(ps_id);
    CHECK(iter != ps_id_to_layers_id_.end()) 
      << "ERROR: cannot find layer for PS id: " << ps_id;
    return ps_layers_[iter->second];
  }
  
  const vector<string>& fc_addrs() { return fc_addrs_; }

  const vector<int>& fc_ids() { return fc_ids_; }

  int num_workers() { return num_workers_; }

  int num_sub_solvers() { return num_sub_solvers_; }

  shared_ptr<Msg> model_server_msg() { return model_server_msg_; }

  string pub_addr() {
    string addr = "tcp://*:";
    addr += boost::lexical_cast<string>(pub_port_);
    
    return addr;
  }

  string router_addr() {
    string addr = "tcp://*:";
    addr += boost::lexical_cast<string>(router_port_);

    return addr;
  }

  int get_staleness() {
    if (model_request_.node_info().has_staleness()) {
      return model_request_.node_info().staleness();
    } else {
      return 0;
    }
  }
  
  int num_splits() { return model_request_.num_splits(); }

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
  

public:
  static inline void set_id_server(const string& addr) {
    id_server_addr_ = addr;
  }

  static inline void set_model_server(const string& addr) {
    model_server_addr_ = addr;
  }
  
  static inline void set_model_request(const ModelRequest& rq) {
    model_request_.CopyFrom(rq);
    has_model_request_ = true;
  }

  static inline void set_request_file(const string& addr) {
    ReadProtoFromTextFileOrDie(addr, &model_request_);
    has_model_request_ = true;
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

  static string RouterAddr(const NodeInfo& node_info) {
    string addr = "tcp://";
    addr += node_info.ip();
    addr += ":";
    addr += boost::lexical_cast<string>(node_info.router_port());
    
    return addr;
  }
  
  static string PubAddr(const NodeInfo& node_info) {
    string addr = "tcp://";
    addr += node_info.ip();
    addr += ":";
    addr += boost::lexical_cast<string>(node_info.pub_port());
    
    return addr;
  }
  
 
protected: 
  static void InitNode(void);
  int InitNodeID();
  int InitModel();
  int InitIP();

protected:
  void InitPSNodes();


private:
  NodeEnv() {
    node_id_ = INVALID_ID;
    num_workers_ = 0;
    num_sub_solvers_ = 0;
  }


protected:
  int node_id_;
  string node_ip_;
  string interface_;

  int router_port_;
  int pub_port_;

  int server_port_;
  
  // sub_solver parameters got from server
  SolverParameter solver_param_;
  RouteInfo rt_info_;
  
  // message got from model server
  shared_ptr<Msg> model_server_msg_;

  // routing addresses

  // addresses of the SUB sockets to receive broadcasting messages
  vector<string> sub_addrs_;

  // address of the dealer sockets to send message to upstream node
  vector<string> prev_router_addrs_;
  
  // ID of the prev nodes, we use dealers to connect upstream nodes
  vector<int> prev_node_ids_;

  // In FC nodes, we broadcast the blobs to downstream nodes
  vector<string> bcast_addrs_;

  // the name of blobs that need to be forwarded
  vector<vector<string> > fwrd_blobs_;

  // addresses for the blobs that need to be forwared
  vector<string> fwrd_addrs_;
  
  // node ID of the nodes
  vector<int> fwrd_ids_;

  // the address of parameter servers
  vector<string> ps_addrs_;

  // the node id of parameter servers
  vector<int> ps_ids_;

  // the ZMQ addresses of all the FC nodes
  vector<string> fc_addrs_;
  
  // address of gateway nodes
  vector<string> gateway_addrs_;

  vector<int> gateway_ids_;

  vector<vector<string> > gateway_blobs_;

  // the node id of FC nodes
  vector<int> fc_ids_;
  
  vector<vector<string> > ps_layers_;

  map<int, int> ps_id_to_layers_id_;

  NodeInfo node_info_;

  // number of input blobs
  int num_input_blobs_;
  
  // mapping message id to solver
  unordered_map<int64_t, void *> id_to_solver_;
  boost::mutex  id_map_mutex_;
  
  // available solvers
  vector<void *> free_solver_;
  
  shared_ptr<SkSock> sk_id_req_;

  int num_workers_;
  
  // number of sub solvers
  int num_sub_solvers_;

private:
  //For Singleton pattern
  static NodeEnv *instance_;
  static boost::once_flag once_;

  // get unique integer id from id server
  static string id_server_addr_;

  static string model_server_addr_;
  
  static bool has_model_request_;

  static ModelRequest model_request_;
  
  static NodeRole node_role_;

DISABLE_COPY_AND_ASSIGN(NodeEnv);
};

} //namespace caffe

#endif


