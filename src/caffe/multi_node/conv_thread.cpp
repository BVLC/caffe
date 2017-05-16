
#include "caffe/multi_node/conv_thread.hpp"

namespace caffe {

template <typename Dtype>
boost::mutex ConvThread<Dtype>::conv_id_mutex_;

template <typename Dtype>
int64_t ConvThread<Dtype>::conv_id_ = 0;

template <typename Dtype>
shared_ptr<Msg> ConvThread<Dtype>::ConvForward()
{
  WorkerSolver<Dtype> *pconv = (WorkerSolver<Dtype> *)NodeEnv::Instance()->PopFreeSolver();
  if (NULL == pconv) {
    pconv = (WorkerSolver<Dtype> *)this->NewSolver();
  }

  shared_ptr<Net<Dtype> > conv_net = pconv->net();
  conv_net->ClearParamDiffs();
  conv_net->ForwardPrefilled();
  
  //copyout the activations
  shared_ptr<Msg> m(new Msg());
  this->CopyOutputData(conv_net, m);
  
  m->set_src(NodeEnv::Instance()->ID());
  int64_t conv_id = NewConvId();
  m->set_type(FORWARD);
  m->set_conv_id(conv_id);
  NodeEnv::Instance()->PutSolver(conv_id, pconv);

  return m;
}

template <typename Dtype>
int ConvThread<Dtype>::ConvBackward(shared_ptr<Msg> m)
{
  WorkerSolver<Dtype> *pconv = (WorkerSolver<Dtype> *)NodeEnv::Instance()->FindSolver(m->conv_id());
  CHECK(pconv != NULL);
  
  shared_ptr<Net<Dtype> > conv_net = pconv->net();
  this->GetOutputDiff(conv_net, m);

  //copy activations
  conv_net->Backward();
  
  //notify the param thread that delta is ready
  shared_ptr<Msg> notify(new Msg(m));
  
  notify->set_dst(ROOT_THREAD_ID);
  notify->set_type(PUT_GRADIENT);
  notify->AppendData(&pconv, sizeof(pconv));
  
  this->SendMsg(notify);

  return 0;
}


template <typename Dtype>
void ConvThread<Dtype>::Run()
{

  while (!this->must_stop()) {
    shared_ptr<Msg> f = ConvForward();
    this->SendMsg(f);

    shared_ptr<Msg> r;
    while ( (r = this->RecvMsg(false)) != NULL) {  //unblocked
    //if ( (r = this->RecvMsg(true)) != NULL) {        //blocked
      ConvBackward(r);      
    }
  }
}


template <typename Dtype>
int ConvParamThread<Dtype>::PutGradient(shared_ptr<Msg> m)
{
  WorkerSolver<Dtype> *psolver = (WorkerSolver<Dtype> *)NodeEnv::Instance()->FindSolver(m->conv_id());
  CHECK(psolver != NULL);
  
  shared_ptr<Msg> ps_msg(new Msg());
  ps_msg->set_type(PUT_GRADIENT);
  ps_msg->set_dst(PS_ID);
  ps_msg->set_src(NodeEnv::Instance()->ID());

  const vector<Blob<Dtype>*>& net_params = psolver->net()->learnable_params();
  for (int i = 0; i < net_params.size(); i++) {
    ps_msg->AppendData(net_params[i]->cpu_diff(), net_params[i]->count() * sizeof(Dtype));
  }
  
  this->SendMsg(ps_msg);
 
  NodeEnv::Instance()->DeleteSolver(m->conv_id());
  NodeEnv::Instance()->PushFreeSolver(psolver);

  return 0;
}

template <typename Dtype>
int ConvParamThread<Dtype>::UpdateParam(shared_ptr<Msg> m)
{
  WorkerSolver<Dtype> *root_solver = (WorkerSolver<Dtype> *) NodeEnv::Instance()->GetRootSolver();
  
  const vector<Blob<Dtype>*>& root_params = root_solver->net()->learnable_params();
  
  CHECK_EQ(root_params.size(), m->ZmsgCnt());

  for (int i = 0; i < root_params.size(); i++) {
    CHECK_EQ(root_params[i]->count() * sizeof(Dtype), m->ZmsgSize(i));

    memcpy(root_params[i]->mutable_cpu_data(), m->ZmsgData(i), m->ZmsgSize(i));
  }

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


