
#include <string>

#include "caffe/multi_node/node_env.hpp"

namespace caffe {

char NodeEnv::id_server_addr_[MAX_STR_LEN];
char NodeEnv::model_server_addr_[MAX_STR_LEN];
ModelRequest NodeEnv::model_request_;
bool NodeEnv::has_model_request_ = false;
NodeRole NodeEnv::node_role_ = INVALID_ROLE;

NodeEnv* NodeEnv::Instance() {
  static NodeEnv node_env;
  return &node_env;
}

void NodeEnv::InitNode(void) {
  // check enviroment settings
  CHECK_GT(strlen(id_server_addr_), 0) << "Need to set id server address";
  CHECK_GT(strlen(model_server_addr_), 0) << "Need to set model server addr";
  CHECK(has_model_request_) << "Need to set model request";
  CHECK(node_role_ != INVALID_ROLE) << "Need to set node role";

  NodeEnv::Instance()->InitIP();
  NodeEnv::Instance()->InitNodeID();
  NodeEnv::Instance()->InitModel();
}

int NodeEnv::InitNodeID() {
  sk_id_req_.reset(new SkSock(ZMQ_REQ));

  sk_id_req_->Connect(string(id_server_addr_));
  shared_ptr<Msg> m(new Msg());

  // adding ip to the message
  m->AppendData(node_ip_.data(), node_ip_.length());
  m->set_type(PING);

  sk_id_req_->SendMsg(m);

  shared_ptr<Msg> r = sk_id_req_->RecvMsg(true);

  int id = *(reinterpret_cast<int *>(r->ZmsgData(0)));

  LOG(INFO) << "Got new id: " << id;

  node_id_ = id;

  return id;
}

int NodeEnv::InitIP() {
  interface_.clear();
  node_ip_.clear();

  if (model_request_.node_info().has_ip()) {
    node_ip_ = model_request_.node_info().ip();
  } else {
    if (model_request_.node_info().has_net_if()) {
      interface_ = model_request_.node_info().net_if();
      node_ip_ = GetIP(interface_);
    } else {
      GetInterfaceAndIP(&interface_, &node_ip_);
    }
  }

  CHECK(!node_ip_.empty()) << "failed to get IP address";
  return 0;
}


void NodeEnv::InitPSNodes() {
  ps_layers_.resize(rt_info_.ps_nodes_size());
  for (int i = 0; i < rt_info_.ps_nodes_size(); i++) {
    const RouteNode& rt_node = rt_info_.ps_nodes(i);
    const NodeInfo& node_info = rt_node.node_info();

    string addr = RouterAddr(node_info);
    ps_addrs_.push_back(addr);
    ps_ids_.push_back(node_info.node_id());

    for (int j = 0; j < rt_node.layers_size(); j++) {
      ps_layers_[i].push_back(rt_node.layers(j));
    }

    ps_id_to_layers_id_[node_info.node_id()] = i;
  }
}

int NodeEnv::InitModel() {
  shared_ptr<SkSock> dealer(new SkSock(ZMQ_DEALER));
  dealer->SetId(node_id_);
  dealer->Connect(string(model_server_addr_));

  // read the request from text file
  ModelRequest rq;
  rq.CopyFrom(model_request_);

  // init route port and pub port
  router_port_ = rq.node_info().router_port();
  pub_port_ = rq.node_info().pub_port();

  rq.mutable_node_info()->set_node_id(node_id_);
  rq.mutable_node_info()->set_ip(node_ip_);
  rq.mutable_node_info()->set_node_role(node_role_);

  node_info_.CopyFrom(rq.node_info());

  string rq_str;
  rq.SerializeToString(&rq_str);

  shared_ptr<Msg> m(new Msg());
  m->AppendData(rq_str.data(), rq_str.length());

  m->set_type(GET_TRAIN_MODEL);
  m->set_src(node_id_);

  LOG(INFO) << "Sending ModelRquest: " << std::endl << rq.DebugString();

  dealer->SendMsg(m);

  shared_ptr<Msg> r = dealer->RecvMsg(true);

  LOG(INFO) << "received message count: " << r->ZmsgCnt();

  rt_info_.ParseFromString(string(reinterpret_cast<char *>(r->ZmsgData(0)),
                           r->ZmsgSize(0) ));
  solver_param_.CopyFrom(rt_info_.solver_param());

  LOG(INFO) << "received route info: " << std::endl << rt_info_.DebugString();
  LOG(INFO) << "received solver: " << std::endl << solver_param_.DebugString();

  for (int i = 0; i < rt_info_.prev_nodes_size(); i++) {
    // dealer addrs
    const NodeInfo& prev_node = rt_info_.prev_nodes(i).node_info();
    if (prev_node.router_port() > 0) {
      string addr = RouterAddr(prev_node);

      prev_router_addrs_.push_back(addr);
      prev_node_ids_.push_back(prev_node.node_id());
    }

    // sub addrs
    if (prev_node.pub_port() > 0) {
      string addr = PubAddr(prev_node);
      sub_addrs_.push_back(addr);
    }
  }


  for (int i = 0; i < rt_info_.bcast_nodes_size(); i++) {
    const NodeInfo& bcast_node = rt_info_.bcast_nodes(i).node_info();
    bcast_addrs_.push_back(bcast_node.ip());
  }

  // add forward node and blobs
  fwrd_blobs_.resize(rt_info_.fwrd_nodes_size());
  for (int i = 0; i < rt_info_.fwrd_nodes_size(); i++) {
    // TODO: add broadcast protocal
    const RouteNode& fwrd_node = rt_info_.fwrd_nodes(i);
    string addr = RouterAddr(fwrd_node.node_info());
    fwrd_addrs_.push_back(addr);

    fwrd_ids_.push_back(fwrd_node.node_info().node_id());
    for (int j = 0; j < fwrd_node.input_blobs_size(); j++) {
      fwrd_blobs_[i].push_back(fwrd_node.input_blobs(j));
    }
  }

  gateway_blobs_.resize(rt_info_.gateway_nodes_size());
  for (int i = 0; i < rt_info_.gateway_nodes_size(); i++) {
    const RouteNode& gw_node = rt_info_.gateway_nodes(i);
    string addr = RouterAddr(gw_node.node_info());
    gateway_addrs_.push_back(addr);
    gateway_ids_.push_back(gw_node.node_info().node_id());
    for (int j = 0; j < gw_node.input_blobs_size(); j++) {
      gateway_blobs_[i].push_back(gw_node.input_blobs(j));
    }
  }

  // add FC nodes if any
  for (int i = 0; i < rt_info_.fc_nodes_size(); i++) {
    const NodeInfo& node = rt_info_.fc_nodes(i).node_info();
    string addr = RouterAddr(node);
    fc_addrs_.push_back(addr);
    fc_ids_.push_back(node.node_id());
  }

  if (rt_info_.has_node_info()) {
    node_info_.CopyFrom(rt_info_.node_info());
  }

  InitPSNodes();

  r->PrintHeader();

  model_server_msg_ = r;

  num_workers_ = rt_info_.num_workers();

  num_sub_solvers_ = rt_info_.num_sub_solvers();

  return 0;
}

}  // end namespace caffe





