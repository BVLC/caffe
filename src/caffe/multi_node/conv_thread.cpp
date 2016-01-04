
#include "caffe/multi_node/conv_thread.hpp"
#include "caffe/multi_node/param_helper.hpp"

namespace caffe {

template <typename Dtype>
boost::mutex ConvThread<Dtype>::conv_id_mutex_;

template <typename Dtype>
int64_t ConvThread<Dtype>::conv_id_ = 0;

template <typename Dtype>
shared_ptr<Msg> ConvThread<Dtype>::ConvForward()
{
  SGDSolver<Dtype> *pconv = (SGDSolver<Dtype> *)NodeEnv::Instance()->PopFreeSolver();
  if (NULL == pconv) {
    Solver<Dtype> *root_solver = (Solver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
    const SolverParameter& solver_param = NodeEnv::Instance()->SolverParam();
    pconv = (SGDSolver<Dtype> *)this->NewSolver(root_solver, solver_param);
  }

  shared_ptr<Net<Dtype> > conv_net = pconv->net();
  conv_net->ClearParamDiffs();
  conv_net->ForwardPrefilled();
  
  //copyout the activations
  shared_ptr<Msg> m(new Msg());
  ParamHelper<Dtype>::CopyOutputDataToMsg(conv_net, m);
  
  m->set_src(NodeEnv::Instance()->ID());
  int64_t conv_id = NewConvId();
  m->set_type(FORWARD);
  m->set_conv_id(conv_id);
  // always use the 0th clock
  m->set_clock(ps_clocks_[0]);
  NodeEnv::Instance()->PutSolver(conv_id, pconv);

  return m;
}

template <typename Dtype>
int ConvThread<Dtype>::ConvBackward(shared_ptr<Msg> m)
{
  int64_t conv_id = m->conv_id();
  SGDSolver<Dtype> *pconv = (SGDSolver<Dtype> *)NodeEnv::Instance()->FindSolver(conv_id);
  CHECK(pconv != NULL);

  shared_ptr<Net<Dtype> > conv_net = pconv->net();
  ParamHelper<Dtype>::CopyOutputDiffFromMsg(conv_net, m);

  conv_net->Backward();
  // pconv->UpdateDiff();

  SGDSolver<Dtype> *root_solver = (SGDSolver<Dtype> *) NodeEnv::Instance()->GetRootSolver();
  ParamHelper<Dtype>::AddDiffFromNet(root_solver->net(), conv_net);
  
  NodeEnv::Instance()->DeleteSolver(conv_id);
  NodeEnv::Instance()->PushFreeSolver(pconv);

  return 0;
}

// TODO: move this to param thread
template <typename Dtype>
void ConvThread<Dtype>::SyncWithPS()
{
  SGDSolver<Dtype> *root_solver = (SGDSolver<Dtype> *) NodeEnv::Instance()->GetRootSolver();
  shared_ptr<Net<Dtype> > conv_net = root_solver->net();

  /// update the clocks to each parameter server
  for (int i = 0; i < ps_clocks_.size(); i++) {
    ps_clocks_[i]++;
  }
  
  /// send the gradient to parameter servers
  for (int i = 0; i < ps_ids_.size(); i++) {
    shared_ptr<Msg> ps_msg(new Msg());
    ps_msg->set_type(PUT_GRADIENT);
    ps_msg->set_dst(ps_ids_[i]);
    ps_msg->set_src(NodeEnv::Instance()->ID());
    ps_msg->set_clock(ps_clocks_[i]);
    
    const vector<string>& ps_layers = NodeEnv::Instance()->FindPSLayer(ps_ids_[i]);
    ParamHelper<Dtype>::CopyParamDiffToMsg(conv_net, ps_layers, ps_msg);
    
    this->SendMsg(ps_msg);
  }
  
  // wait for the response from parameter servers
  int num_param_update = 0;
  shared_ptr<Msg> m;
  while ((m = this->RecvMsg(true)) != NULL) {
    if (m->type() == PUT_PARAM) {
      map<int, int>::iterator map_iter = ps_id_map_.find(m->src());
      CHECK(map_iter != ps_id_map_.end());

      ParamHelper<Dtype>::CopyParamDataFromMsg(conv_net, m);
      
      num_param_update++;
      if (num_param_update >= ps_ids_.size()) {
        break;
      }
    }
  }
}

template <typename Dtype>
void ConvThread<Dtype>::Run()
{
  while (!this->must_stop()) {
    for (int i = 0; i < NUM_SUB_SOLVERS; i++) {
      shared_ptr<Msg> f = ConvForward();
    
      #if 0
      int msg_size = 0;
      for (int i = 0; i < f->ZmsgCnt(); i++) {
        msg_size += f->ZmsgSize(i);
      }
      LOG(INFO) << "message size: " << msg_size;
      #endif

      this->SendMsg(f);
    }

    SGDSolver<Dtype> *root_solver = (SGDSolver<Dtype> *) NodeEnv::Instance()->GetRootSolver();
    root_solver->net()->ClearParamDiffs();
    
    for (int i = 0; i < NUM_SUB_SOLVERS; i++) {
      shared_ptr<Msg> r;
      //while ( (r = this->RecvMsg(false)) != NULL) {  //unblocked
      if ( (r = this->RecvMsg(true)) != NULL) {        //blocked
        ConvBackward(r);
      }
    }

    SyncWithPS();
  }

}


template <typename Dtype>
int ConvParamThread<Dtype>::PutGradient(shared_ptr<Msg> m)
{
  WorkerSolver<Dtype> *psolver = (WorkerSolver<Dtype> *)NodeEnv::Instance()->FindSolver(m->conv_id());
  CHECK(psolver != NULL);
  
  shared_ptr<Net<Dtype> > net = psolver->net();
  
  /// send the gradient to parameter servers
  for (int i = 0; i < ps_ids_.size(); i++) {
    shared_ptr<Msg> ps_msg(new Msg());
    ps_msg->set_type(PUT_GRADIENT);
    ps_msg->set_dst(ps_ids_[i]);
    ps_msg->set_src(NodeEnv::Instance()->ID());
    
    const vector<string>& ps_layers = NodeEnv::Instance()->FindPSLayer(ps_ids_[i]);
    ParamHelper<Dtype>::CopyParamDiffToMsg(net, ps_layers, ps_msg);
    
    this->SendMsg(ps_msg);
  }

  NodeEnv::Instance()->DeleteSolver(m->conv_id());
  NodeEnv::Instance()->PushFreeSolver(psolver);

  return 0;
}

template <typename Dtype>
int ConvParamThread<Dtype>::UpdateParam(shared_ptr<Msg> m)
{
  SGDSolver<Dtype> *root_solver = (SGDSolver<Dtype> *) NodeEnv::Instance()->GetRootSolver();
  
  /// TODO: add flow control when computation is faster than PS communication
  ParamHelper<Dtype>::CopyParamDataFromMsg(root_solver->net(), m);
  
  return 0;
}

template <typename Dtype>
void ConvParamThread<Dtype>::Run()
{
  Caffe::set_root_solver(true);

  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);

    if (m->type() == PUT_GRADIENT) {
      PutGradient(m);
    } else if (m->type() == PUT_PARAM) {
      UpdateParam(m);
    } else {
      LOG(ERROR) << "PS client: unknown type " << m->type();
    }
  }
  
}

INSTANTIATE_CLASS(ConvThread);
INSTANTIATE_CLASS(ConvParamThread);

} // end namespace caffe


