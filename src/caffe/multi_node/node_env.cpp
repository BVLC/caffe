

#include "caffe/multi_node/node_env.hpp"

namespace caffe {

NodeEnv *NodeEnv::instance_ = NULL;
boost::once_flag NodeEnv::once_;
string NodeEnv::id_server_addr_;
string NodeEnv::model_server_addr_;
ModelRequest NodeEnv::model_request_;
bool NodeEnv::has_model_request_ = false;
NodeRole NodeEnv::node_role_ = INVALID_ROLE;

NodeEnv* NodeEnv::Instance()
{
  boost::call_once(once_, InitNode);

  return instance_;
}

void NodeEnv::InitNode(void)
{
  // check enviroment settings
  CHECK(!id_server_addr_.empty()) << "Need to set id server address before start";
  CHECK(!model_server_addr_.empty()) << "Need to set model server address before start";
  CHECK(has_model_request_) << "Need to set model request firstly";
  CHECK(node_role_ != INVALID_ROLE) << "Need to set node role before start";

  instance_ = new NodeEnv();
  
  instance_->InitIP();
  instance_->InitNodeID();
  instance_->InitModel();
}

int NodeEnv::InitNodeID()
{
  shared_ptr<SkSock> req(new SkSock(ZMQ_REQ));

  req->Connect(id_server_addr_);
  shared_ptr<Msg> m(new Msg());
  
  //adding ip to the message
  m->AppendData(node_ip_.data(), node_ip_.length());
  m->set_type(PING);

  req->SendMsg(m);

  shared_ptr<Msg> r = req->RecvMsg(true);
  
  int id = *((int *)r->ZmsgData(0));

  LOG(INFO) << "Got new id: " << id;

  node_id_ = id;

  return id;
}

int NodeEnv::InitIP()
{
  interface_.clear();
  node_ip_.clear();

  GetInterfaceAndIP(interface_, node_ip_);

  return 0;
}


void NodeEnv::InitPSNodes()
{
  ps_layers_.resize(rt_info_.ps_nodes_size());
  for (int i = 0; i < rt_info_.ps_nodes_size(); i++) {
    const NodeInfo& node_info = rt_info_.ps_nodes(i);
    
    string addr = RouterAddr(node_info);
    ps_addrs_.push_back(addr);
    ps_ids_.push_back(node_info.node_id());

    for (int j = 0; j < node_info.layers_size(); j++) {
      ps_layers_[i].push_back(node_info.layers(j));
    }
    
    ps_id_to_layers_id_[node_info.node_id()] = i;
  }
}

int NodeEnv::InitModel()
{
  shared_ptr<SkSock> dealer(new SkSock(ZMQ_DEALER));
  dealer->SetId(node_id_);
  dealer->Connect(model_server_addr_);
  
  //read the request from text file
  ModelRequest rq;
  rq.CopyFrom(model_request_);

  //init route port and pub port
  router_port_ = rq.node_info().router_port();
  pub_port_ = rq.node_info().pub_port();
  
  rq.mutable_node_info()->set_node_id(node_id_);
  rq.mutable_node_info()->set_ip(node_ip_);
  rq.mutable_node_info()->set_node_role(node_role_);

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
  
  
  rt_info_.ParseFromString(string( (char *)r->ZmsgData(0), r->ZmsgSize(0) ));
  solver_param_.CopyFrom(rt_info_.solver_param());

  LOG(INFO) << "received route info: " << std::endl << rt_info_.DebugString();
  LOG(INFO) << "received solver: " << std::endl << solver_param_.DebugString();
  
  for (int i = 0; i < rt_info_.prev_nodes_size(); i++) {
    //dealer addrs
    if (rt_info_.prev_nodes(i).router_port() > 0) {
      string addr = RouterAddr(rt_info_.prev_nodes(i));
      
      prev_router_addrs_.push_back(addr);
      prev_node_ids_.push_back(rt_info_.prev_nodes(i).node_id());
    }

    //sub addrs
    if (rt_info_.prev_nodes(i).pub_port() > 0) {
      string addr = PubAddr(rt_info_.prev_nodes(i)); 
      sub_addrs_.push_back(addr);
    }
  }
  
  
  for (int i = 0; i < rt_info_.bcast_nodes_size(); i++) {
    bcast_addrs_.push_back(rt_info_.bcast_nodes(i).ip());
  }

  // add forward node and blobs
  for (int i = 0; i < rt_info_.fwrd_nodes_size(); i++) {
    // TODO: add broadcast protocal
    string addr = RouterAddr(rt_info_.fwrd_nodes(i));
    fwrd_addrs_.push_back(addr);

    fwrd_ids_.push_back(rt_info_.fwrd_nodes(i).node_id());
    fwrd_blobs_.push_back(rt_info_.fwrd_blob(i));
  }
  
  // add FC gateway addr
  if (rt_info_.has_gateway_node()) {
    fc_gateway_addr_ = "tcp://";
    fc_gateway_addr_ += rt_info_.gateway_node().ip();
    fc_gateway_addr_ += ":";
    fc_gateway_addr_ += boost::lexical_cast<string>(GATEWAY_PORT);
  }
  
  // add FC nodes if any
  for (int i = 0; i < rt_info_.fc_nodes_size(); i++) {
    const NodeInfo& node = rt_info_.fc_nodes(i);
    string addr = RouterAddr(node);
    fc_addrs_.push_back(addr);
    fc_ids_.push_back(node.node_id());
  }

  InitPSNodes();

  r->PrintHeader();

  model_server_msg_ = r;

  return 0;
}

} //end caffe





