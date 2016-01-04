
#include "caffe/multi_node/fc_thread.hpp"
#include "caffe/multi_node/param_helper.hpp"

namespace caffe {

template <typename Dtype>
shared_ptr<Msg> FcThread<Dtype>::FcForward(shared_ptr<Msg> m)
{
  SGDSolver<Dtype> *pfc = (SGDSolver<Dtype> *)NodeEnv::Instance()->PopFreeSolver();
  Solver<Dtype> *proot = (Solver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
  
  if (NULL == pfc) {
    const SolverParameter& solver_param = NodeEnv::Instance()->SolverParam();

    pfc = (SGDSolver<Dtype> *)this->NewSolver(proot, solver_param);
  }
  
  if (!ParamHelper<Dtype>::IsParamShared(pfc->net(), proot->net())) {
    // copy param data from root solver
    const vector<Blob<Dtype>*>& params = pfc->net()->learnable_params();
    const vector<Blob<Dtype>*>& root_params = proot->net()->learnable_params();
    CHECK_EQ(params.size(), root_params.size());
    
    for (int i = 0; i < params.size(); i++) {
      CHECK_EQ(params[i]->count(), root_params[i]->count());
      memcpy(params[i]->mutable_cpu_data(), root_params[i]->cpu_data(), root_params[i]->count() * sizeof(Dtype));
    }
  }
 
  shared_ptr<Net<Dtype> > fc_net = pfc->net();
  fc_net->ClearParamDiffs();

  ParamHelper<Dtype>::CopyInputDataFromMsg(fc_net, m);
  fc_net->ForwardPrefilled();
  
  shared_ptr<Msg> r(new Msg(m));
  //broadcast the message
  r->set_dst(-1);
  ParamHelper<Dtype>::CopyOutputDataToMsg(fc_net, r);

  NodeEnv::Instance()->PutSolver(m->msg_id(), pfc);

  return r;
}

template <typename Dtype>
void FcThread<Dtype>::FcBackward(shared_ptr<Msg> m, vector<shared_ptr<Msg> >& replies, bool copy_diff)
{
  SGDSolver<Dtype> *pfc = (SGDSolver<Dtype> *)NodeEnv::Instance()->FindSolver(m->msg_id());
  CHECK(pfc != NULL);

  shared_ptr<Net<Dtype> > fc_net = pfc->net();
  if (copy_diff) {
    ParamHelper<Dtype>::CopyOutputDiffFromMsg(fc_net, m);
  }

  fc_net->Backward();

  const vector<int>& pre_ids = NodeEnv::Instance()->prev_node_ids();

  if (pre_ids.size() == 0) { //we are the gateway node
    shared_ptr<Msg> r(new Msg(m));
    r->set_dst(m->src());
    replies.push_back(r);
  } else if (pre_ids.size() > 0) {
    for (int i = 0; i < pre_ids.size(); i++) {
      shared_ptr<Msg> r(new Msg(m));
      r->set_dst(pre_ids[i]);
      replies.push_back(r);
    }
  } else {
    //array size cannot be less than 0
  }
  
  //copy diff to downstream nodes
  for (int i = 0; i < replies.size(); i++) {
    shared_ptr<Msg> r = replies[i];

    r->set_type(BACKWARD);
    ParamHelper<Dtype>::CopyInputDiffToMsg(fc_net, r);
  }
  
  // pfc->UpdateDiff();
  
  //notify the param thread
  shared_ptr<Msg> notify(new Msg(m));

  notify->set_dst(ROOT_THREAD_ID);
  notify->AppendData(&pfc, sizeof(pfc));
  replies.push_back(notify);
}

template <typename Dtype>
void FcThread<Dtype>::Run()
{
  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);
    
    vector<shared_ptr<Msg> > msg_arr;

    if (m->type() == FORWARD) {
      shared_ptr<Msg> f = FcForward(m);
      msg_arr.push_back(f);
    } else if (m->type() == BACKWARD) {
      FcBackward(m, msg_arr, true);
    } else {
      LOG(INFO) << "unkown type: " << m->msg_id();
    }
    
    for (int i = 0; i < msg_arr.size(); i++) {
      this->SendMsg(msg_arr[i]);
    }
  }
}

template <typename Dtype>
boost::atomic_int FcLossThread<Dtype>::iter_(0);

template <typename Dtype>
void FcLossThread<Dtype>::Run()
{
  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);
    
    shared_ptr<Msg> f = this->FcForward(m);
    
    vector<shared_ptr<Msg> > replies;
    this->FcBackward(f, replies, false);

    iter_++;
    Dtype loss = *((Dtype *)f->ZmsgData(0));
    LOG(INFO) << "train iteration: " << iter_ << " loss: " << loss;

    for (int i = 0; i < replies.size(); i++) {
      this->SendMsg(replies[i]);
    }
  }
}

template <typename Dtype>
void FcParamThread<Dtype>::UpdateParam(shared_ptr<Msg> m)
{
  SGDSolver<Dtype> *psolver = (SGDSolver<Dtype> *)NodeEnv::Instance()->FindSolver(m->msg_id());
  CHECK(psolver != NULL);

  SGDSolver<Dtype> *proot = (SGDSolver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
  
  #if 0
  map<int, int>::iterator map_iter = client_idx_map_.find(m->src());
  int client_idx = -1;
  if (map_iter == client_idx_map_.end()) {
    // add new client
    client_idx = AddNewClient(m, psolver);
  } else {
    client_idx = map_iter->second;
    client_clocks_[client_idx] = m->clock();
    // TODO: allow one client has many solvers
    CHECK(client_solvers_[client_idx] == NULL);
    client_solvers_[client_idx] = psolver;
    msg_ids_[client_idx] = m->msg_id();
  }
  
  int clock_bound = MinClock() + staleness_;
  int num_updates = 0;
  // update net diff
  for (int i = 0; i < client_ids_.size(); i++) {
    /// LOG(INFO) << "client " << client_ids_[i] << " clock: " << client_clocks_[i];
    if (client_clocks_[i] <= clock_bound && client_solvers_[i] != NULL) {
      num_updates++;
      ParamHelper<Dtype>::AddDiffFromNet(proot->net(), client_solvers_[i]->net());
      NodeEnv::Instance()->DeleteSolver(msg_ids_[i]);
      NodeEnv::Instance()->PushFreeSolver(client_solvers_[i]);
      client_solvers_[i] = NULL;
      // LOG(INFO) << "update client: " << client_ids_[i];
    }
  }
  
  if (num_updates > 0) {
    proot->net()->Update();
    proot->net()->ClearParamDiffs();
    train_iter_++;
  }
  #endif
  
  if (sub_updates_ == 0) {
    ParamHelper<Dtype>::CopyDiffFromNet(proot->net(), psolver->net());
  } else {
    ParamHelper<Dtype>::AddDiffFromNet(proot->net(), psolver->net());
  }
  
  sub_updates_++;
  
  if (sub_updates_ >= NUM_SUB_SOLVERS) {
    proot->CommitGradient();
    sub_updates_ = 0;
  }
  
  NodeEnv::Instance()->DeleteSolver(m->msg_id());
  NodeEnv::Instance()->PushFreeSolver(psolver);
  train_iter_++;

  // doesn't have next nodes means we have loss layer
  const vector<string>& bcast_addrs = NodeEnv::Instance()->bcast_addrs();

  if (test_node_id_ > 0  
     && train_iter_ % TRAIN_NOTIFY_INTERVAL == 0
     &&  bcast_addrs.size() == 0
      ) {
    this->SendNotify();
  }
}

template <typename Dtype>
void FcParamThread<Dtype>::SendNotify()
{
  shared_ptr<Msg> r(new Msg());
  r->set_type(TRAIN_ITER);
  r->set_dst(test_node_id_);
  r->set_src(NodeEnv::Instance()->ID());
  
  r->AppendData(&train_iter_, sizeof(train_iter_));
  
  // LOG(INFO) << "sending notify";

  this->SendMsg(r);
}


template <typename Dtype>
void FcParamThread<Dtype>::SendParam(shared_ptr<Msg> m)
{
  shared_ptr<Msg> r(new Msg());
  r->set_type(PUT_PARAM);
  r->set_dst(m->src());
  r->set_src(NodeEnv::Instance()->ID());
  
  Solver<Dtype> *psolver = (Solver<Dtype> *)NodeEnv::Instance()->GetRootSolver();

  shared_ptr<Net<Dtype> > net = psolver->net();
  ParamHelper<Dtype>::CopyParamDataToMsg(net, net->layer_names(), r);
  
  // LOG(INFO) << "sending param";

  this->SendMsg(r);
}

template <typename Dtype>
void FcParamThread<Dtype>::Run()
{
  //use the root solver
  Caffe::set_root_solver(true);

  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);
    
    if (m->type() == GET_PARAM) {
      test_node_id_ = m->src();
      SendParam(m);
    } else {
      UpdateParam(m);
    }
  }
}

INSTANTIATE_CLASS(FcThread);
INSTANTIATE_CLASS(FcLossThread);
INSTANTIATE_CLASS(FcParamThread);


} //end caffe


