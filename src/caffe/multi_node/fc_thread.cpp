
#include "caffe/multi_node/fc_thread.hpp"

namespace caffe {

template <typename Dtype>
shared_ptr<Msg> FcThread<Dtype>::FcForward(shared_ptr<Msg> m)
{
  SGDSolver<Dtype> *pfc = (SGDSolver<Dtype> *)NodeEnv::Instance()->PopFreeSolver();
  if (NULL == pfc) {
    pfc = (SGDSolver<Dtype> *)this->NewSolver();
  }
  
  shared_ptr<Net<Dtype> > fc_net = pfc->net();
  fc_net->ClearParamDiffs();

  this->GetInputData(fc_net, m);
  fc_net->ForwardPrefilled();
  
  shared_ptr<Msg> r(new Msg(m));
  //broadcast the message
  r->set_dst(-1);
  this->CopyOutputData(fc_net, r);

  NodeEnv::Instance()->PutSolver(m->msg_id(), pfc);

  return r;
}

template <typename Dtype>
void FcThread<Dtype>::FcBackward(shared_ptr<Msg> m, vector<shared_ptr<Msg> >& replies)
{
  SGDSolver<Dtype> *pfc = (SGDSolver<Dtype> *)NodeEnv::Instance()->FindSolver(m->msg_id());
  CHECK(pfc != NULL);

  shared_ptr<Net<Dtype> > fc_net = pfc->net();
  this->GetOutputDiff(fc_net, m);

  fc_net->Backward();

  const vector<int>& pre_ids = NodeEnv::Instance()->DealerIDs();

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
  
  //copy data
  for (int i = 0; i < replies.size(); i++) {
    shared_ptr<Msg> r = replies[i];

    r->set_type(BACKWARD);
    this->CopyInputDiff(fc_net, r);
  }
  
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

    LOG(INFO) << "received message id: " << m->msg_id() << " with blobs: " << m->num_blobs();
    
    vector<shared_ptr<Msg> > msg_arr;

    if (m->type() == FORWARD) {
      shared_ptr<Msg> f = FcForward(m);
      msg_arr.push_back(f);
    } else if (m->type() == BACKWARD) {
      LOG(INFO) << "received backward message";

      FcBackward(m, msg_arr);
    } else {
      LOG(INFO) << "unkown type: " << m->msg_id();
    }
    
    for (int i = 0; i < msg_arr.size(); i++) {
      this->SendMsg(msg_arr[i]);
    }
  }
}

template <typename Dtype>
boost::atomic_int FcEndThread<Dtype>::iter_(0);

template <typename Dtype>
void FcEndThread<Dtype>::Run()
{
  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);
    
    shared_ptr<Msg> f = this->FcForward(m);
    
    vector<shared_ptr<Msg> > replies;
    this->FcBackward(f, replies);
    
    iter_++;
    Dtype loss = *((Dtype *)f->ZmsgData(0));
    LOG(INFO) << "iteration: " << iter_ << " loss: " << loss;

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

  psolver->CommitGradient();
  
  //update table
  NodeEnv::Instance()->DeleteSolver(m->msg_id());
  NodeEnv::Instance()->PushFreeSolver(psolver);
}

template <typename Dtype>
void FcParamThread<Dtype>::Run()
{
  //use the root solver
  Caffe::set_root_solver(true);

  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);
    
    UpdateParam(m);
  }
}

INSTANTIATE_CLASS(FcThread);
INSTANTIATE_CLASS(FcEndThread);
INSTANTIATE_CLASS(FcParamThread);

} //end caffe


