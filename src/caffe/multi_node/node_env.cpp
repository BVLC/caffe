

#include "caffe/multi_node/node_env.hpp"

namespace caffe {

NodeEnv *NodeEnv::instance_ = NULL;
boost::once_flag NodeEnv::once_;
string NodeEnv::id_server_addr_;
string NodeEnv::model_server_addr_;
string NodeEnv::request_file_addr_;
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
  CHECK(!request_file_addr_.empty()) << "Need to set the location of model request file";
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



int NodeEnv::InitModel()
{
  shared_ptr<SkSock> dealer(new SkSock(ZMQ_DEALER));
  dealer->SetId(node_id_);
  dealer->Connect(model_server_addr_);
  
  //read the request from text file
  ModelRequest rq;
  ReadProtoFromTextFileOrDie(request_file_addr_, &rq);
  
  //init route port and pub port
  router_port_ = rq.node_info().router_port();
  pub_port_ = rq.node_info().pub_port();
  

  rq.mutable_node_info()->set_node_id(node_id_);
  rq.mutable_node_info()->set_ip(node_ip_); 

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
  solver_param_.ParseFromString(string( (char *) r->ZmsgData(1), r->ZmsgSize(1) ));

  LOG(INFO) << "received route info: " << std::endl << rt_info_.DebugString();
  LOG(INFO) << "received solver: " << std::endl << solver_param_.DebugString();
  
  for (int i = 0; i < rt_info_.prev_nodes_size(); i++) {
    //dealer addrs
    if (rt_info_.prev_nodes(i).router_port() > 0) {
      string addr = "tcp://";
      addr += rt_info_.prev_nodes(i).ip();
      addr += ":";
      addr += boost::lexical_cast<string>(rt_info_.prev_nodes(i).router_port());

      dealer_addrs_.push_back(addr);
      dealer_ids_.push_back(rt_info_.prev_nodes(i).node_id());
    }

    //sub addrs
    if (rt_info_.prev_nodes(i).pub_port() > 0) {
      string addr = "tcp://";
      addr += rt_info_.prev_nodes(i).ip();
      addr += ":";
      addr += boost::lexical_cast<string>(rt_info_.prev_nodes(i).pub_port());

      sub_addrs_.push_back(addr);
    }
  }
  
  
  for (int i = 0; i < rt_info_.next_nodes_size(); i++) {
    //we don't add port to the string
    client_addrs_.push_back(rt_info_.next_nodes(i).ip());
  }

  for (int i = 0; i < rt_info_.bottom_nodes_size(); i++) {
    //
    string addr = "tcp://";
    addr += rt_info_.bottom_nodes(i).ip();
    addr += ":";
    addr += boost::lexical_cast<string>(rt_info_.bottom_nodes(i).router_port());

    bottom_addrs_.push_back(addr);
    bottom_ids_.push_back(rt_info_.bottom_nodes(i).node_id());
  }
  
  num_input_blobs_ = 0;

  if (solver_param_.has_net_param()) {
    num_input_blobs_ = solver_param_.net_param().input_size();
  }

  return 0;
}

} //end caffe





