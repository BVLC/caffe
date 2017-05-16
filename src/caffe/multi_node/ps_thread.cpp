
#include "caffe/multi_node/ps_thread.hpp"

namespace caffe {

template <typename Dtype>
void PSThread<Dtype>::UpdateParam(shared_ptr<Msg> m)
{
  const vector<Blob<Dtype>*>& net_params = ps_solver_->net()->learnable_params();

  CHECK_EQ(net_params.size(), m->ZmsgCnt());

  for (int i = 0; i < net_params.size(); i++) {
      CHECK_EQ(net_params[i]->count() * sizeof(Dtype), m->ZmsgSize(i));
      memcpy(net_params[i]->mutable_cpu_diff(), m->ZmsgData(i), m->ZmsgSize(i));
  }

  ps_solver_->CommitGradient();
}


template <typename Dtype>
void PSThread<Dtype>::SendParam(shared_ptr<Msg> m)
{
  const vector<Blob<Dtype>*>& net_params = ps_solver_->net()->learnable_params();

  shared_ptr<Msg> r(new Msg());
  r->set_type(PUT_PARAM);
  r->set_dst(m->src());
  r->set_src(NodeEnv::Instance()->ID());

  for (int i = 0; i < net_params.size(); i++) {
    r->AppendData(net_params[i]->cpu_data(), net_params[i]->count() * sizeof(Dtype));
  }

  this->SendMsg(r);
}


template <typename Dtype>
void PSThread<Dtype>::Run()
{
  Caffe::set_root_solver(false);
  //create a solver that shares data params with root solver
  ps_solver_ = (SGDSolver<Dtype> *)this->NewSolver();
  Caffe::set_root_solver(true);

  while (true) {
    shared_ptr<Msg> m = this->RecvMsg(true);

    if (m->type() == PUT_GRADIENT) {
      UpdateParam(m);
      SendParam(m);
    } else if (m->type() == GET_PARAM) {
      SendParam(m);
    } else {
      LOG(ERROR) << "Cannot deal with message type: " << m->type() 
        << " from: " << m->src();
    }

  }
}

INSTANTIATE_CLASS(PSThread);
} //end caffe



