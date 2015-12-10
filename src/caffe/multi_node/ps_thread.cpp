
#include "caffe/multi_node/ps_thread.hpp"
#include "caffe/multi_node/param_helper.hpp"
#include <boost/make_shared.hpp>

namespace caffe {

template <typename Dtype>
void PSThread<Dtype>::UpdateParam(shared_ptr<Msg> m)
{
  map<int, int>::iterator map_iter = client_idx_map_.find(m->src());
  
  if (map_iter == client_idx_map_.end()) {
    LOG(WARNING) << "update from unregistered node id: " << m->src();
    return;
  }

  int client_idx = map_iter->second;
  client_need_update_[client_idx] = true;
  client_clocks_[client_idx]++;
  client_msgs_[client_idx] = m;

  if (client_clocks_[client_idx] != m->clock()) {
    LOG(INFO) << "unmatched clock between ps and client " << m->src();
    client_clocks_[client_idx] = m->clock();
  }

  SendUpdates();
}


template <typename Dtype>
void PSThread<Dtype>::SendParam(int dst, int clock)
{
  shared_ptr<Msg> r(new Msg());
  r->set_type(PUT_PARAM);
  r->set_dst(dst);
  r->set_src(NodeEnv::Instance()->ID());
  r->set_clock(clock);

  shared_ptr<Net<Dtype> > net = ps_solver_->net();
  ParamHelper<Dtype>::CopyParamDataToMsg(net, net->layer_names(), r);
  
  this->SendMsg(r);
}


template <typename Dtype>
void PSThread<Dtype>::SendUpdates()
{
  if (staleness_ < 0) {
    for (int i = 0; i < client_ids_.size(); i++) {
      if (client_need_update_[i]) {
        SendParam(client_ids_[i], client_clocks_[i]);
        // mark we are waiting for clients to join
        client_need_update_[i] = false;
      }
    }
  } else {
    int clock_bound = MinClock() + staleness_;
    
    int net_updates = 0;
    // update parameters from clients
    for (int i = 0; i < client_msgs_.size(); i++) {
      if (client_msgs_[i] != NULL && client_clocks_[i] <= clock_bound) {
        ParamHelper<Dtype>::AddDiffFromMsg(ps_solver_->net(), client_msgs_[i]);
        client_msgs_[i].reset();
        net_updates++;
      }
    }

    if (net_updates > 0) {
      ps_solver_->net()->Update();
      ps_solver_->net()->ClearParamDiffs();
      iter_++;
    }

    // send the parameters to clients
    for (int i = 0; i < client_ids_.size(); i++) {
      if (client_need_update_[i] && client_clocks_[i] <= clock_bound) {
        SendParam(client_ids_[i], client_clocks_[i]);
        // mark we are waiting for clients to join
        client_need_update_[i] = false;
      }
    }
  }
}

template <typename Dtype>
int PSThread<Dtype>::MinClock()
{
  // clock starts from 0
  int min_clock = 0;

  if (client_clocks_.size() > 0) {
    min_clock = client_clocks_[0];
    
    for (int i = 1; i < client_clocks_.size(); i++) {
      if (client_clocks_[i] < min_clock) {
        min_clock = client_clocks_[i];
      }
    }
  }

  return min_clock;
}

template <typename Dtype>
void PSThread<Dtype>::RegisterNode(shared_ptr<Msg> m)
{
  LOG(INFO) << "registering node: " << m->src();
  
  //init clock as the minimal clock among the clients
  int clock = MinClock() + staleness_ + 1;
  client_clocks_.push_back(clock);
  
  int client_idx = client_ids_.size();

  int node_id = m->src();
  client_ids_.push_back(node_id);
  client_idx_map_[node_id] = client_idx;

  // send parameters to the client by default
  client_need_update_.push_back(true);
  // push back a null msg for init
  shared_ptr<Msg> null_msg;
  client_msgs_.push_back(null_msg);

  SendUpdates();
}

template <typename Dtype>
void PSThread<Dtype>::Run()
{
  Caffe::set_root_solver(false);
  // use root solver as ps solver
  ps_solver_ = (SGDSolver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
  Caffe::set_root_solver(true);
  
  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);

    if (m->type() == PUT_GRADIENT) {
      UpdateParam(m);
    } else if (m->type() == REGISTER_NODE) {
      RegisterNode(m);
    } else if (m->type() == GET_PARAM) {
      SendParam(m->src(), MinClock());
    } else {
      LOG(ERROR) << "Cannot deal with message type: " << m->type() 
        << " from: " << m->src();
    }
  }
}

INSTANTIATE_CLASS(PSThread);
} //end caffe



