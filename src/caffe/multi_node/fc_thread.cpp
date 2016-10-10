
#include <boost/make_shared.hpp>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/multi_node/fc_thread.hpp"
#include "caffe/multi_node/param_helper.hpp"

namespace caffe {

template <typename Dtype>
ParamBuf<Dtype> *FcWorker<Dtype>::pbuf_ = NULL;

template <typename Dtype>
boost::once_flag FcWorker<Dtype>::once_;

template <typename Dtype>
vector<Blob<Dtype>*> * ParamBuf<Dtype>::RefParam(void *psolver, int clock) {
  boost::mutex::scoped_lock rlock(ref_mutex_);

  psolver_to_clock_[psolver] = clock;
  unordered_map<int, int>::iterator clock_iter = clock_to_idx_.find(clock);

  int idx = -1;
  if (clock_iter != clock_to_idx_.end()) {
    // use the param associated with the clock
    idx = clock_iter->second;
  } else {
    // use the latest param for this clock
    unordered_map<void *, int>::iterator iter =
                                      pointer_to_idx_.find(platest_param_);
    CHECK(iter != pointer_to_idx_.end()) << "cannot find index to pointer: "
                                         << platest_param_;
    idx = iter->second;
    clock_to_idx_[clock] = idx;
  }

  psolver_to_idx_[psolver] = idx;
  ref_cnt_vec_[idx]++;

  return param_vec_[idx];
}

template <typename Dtype>
vector<Blob<Dtype>*> *ParamBuf<Dtype>::FindParam(void *psolver) {
  boost::mutex::scoped_lock lock(ref_mutex_);

  unordered_map<void *, int>::iterator iter = psolver_to_idx_.find(psolver);
  CHECK(iter != psolver_to_idx_.end()) << "cannot find index to pointer: "
                                       << psolver;

  int idx = iter->second;

  return param_vec_[idx];
}


template <typename Dtype>
int ParamBuf<Dtype>::DeRefParam(void *psolver) {
  boost::mutex::scoped_lock lock(ref_mutex_);

  unordered_map<void *, int>::iterator iter = psolver_to_idx_.find(psolver);
  CHECK(iter != psolver_to_idx_.end()) << "cannot find index to pointer: "
                                       << psolver;

  int idx = iter->second;

  psolver_to_idx_.erase(iter);
  ref_cnt_vec_[idx]--;
  CHECK_GE(ref_cnt_vec_[idx], 0) << "unexpected reference counter";

  unordered_map<void *, int>::iterator clock_iter =
                                          psolver_to_clock_.find(psolver);
  CHECK(clock_iter != psolver_to_clock_.end());

  psolver_to_clock_.erase(clock_iter);

  return ref_cnt_vec_[idx];
}


template <typename Dtype>
vector<Blob<Dtype>*> *ParamBuf<Dtype>::CreateParam(
                            const vector<Blob<Dtype>*> &params) {
  vector<Blob<Dtype>*> *pblobs = new vector<Blob<Dtype>*>();

  for (int i = 0; i < params.size(); i++) {
    Blob<Dtype>* pb = new Blob<Dtype>();
    pb->ReshapeLike(*params[i]);

    pblobs->push_back(pb);
  }

  // insert the paramter to buffer
  boost::mutex::scoped_lock lock(ref_mutex_);

  pointer_to_idx_[pblobs] = param_vec_.size();
  param_vec_.push_back(pblobs);
  ref_cnt_vec_.push_back(0);

  CHECK_EQ(ref_cnt_vec_.size(), param_vec_.size());

  LOG(INFO) << "created " << ref_cnt_vec_.size() << " parameters";

  return pblobs;
}

template <typename Dtype>
void ParamBuf<Dtype>::InitParamBuf(const vector<Blob<Dtype>*> &params) {
  // create 4 paramters in the beginning
  for (int i = 0; i < 4; i++) {
    CreateParam(params);
  }
}

template <typename Dtype>
vector<Blob<Dtype>*> *ParamBuf<Dtype>::GetParam() {
  return platest_param_;
}

template <typename Dtype>
vector<Blob<Dtype>*> *ParamBuf<Dtype>::FindFreeParam() {
  boost::mutex::scoped_lock lock(ref_mutex_);

  // find a free param pointer
  vector<Blob<Dtype>*> *pfree = NULL;
  for (int i = 0; i < param_vec_.size(); i++) {
    if (ref_cnt_vec_[i] == 0 && param_vec_[i] != platest_param_) {
      pfree = param_vec_[i];
    }
  }

  return pfree;
}

template <typename Dtype>
void ParamBuf<Dtype>::ReplaceParam(vector<Blob<Dtype>*> *p) {
  boost::mutex::scoped_lock lock(ref_mutex_);
  platest_param_ = p;
}

template <typename Dtype>
void FcThread<Dtype>::CopyInputDataFromMsg(shared_ptr<Net<Dtype> > fc_net,
                                           shared_ptr<Msg> m) {
  for (int i = 0; i < fc_net->num_inputs(); i++) {
    int blob_index = fc_net->input_blob_indices()[i];
    const string& blob_name = fc_net->blob_names()[blob_index];

    Blob<Dtype>* pblob = fc_net->input_blobs()[i];
    ParamHelper<Dtype>::CopyBlobDataFromMsg(pblob, blob_name, m);
  }
}


template <typename Dtype>
void FcThread<Dtype>::FcForward(shared_ptr<Msg> m) {
  MLOG(INFO) << "Begin forward for src: " << m->src()
             << ", ID: " << m->conv_id();

  group_iter_t grp_iter = clock_to_solver_grp_.find(m->clock());

  shared_ptr<SolverGroup<Dtype> > pgrp;
  if (grp_iter == clock_to_solver_grp_.end()) {
    pgrp = boost::make_shared<SolverGroup<Dtype> >(m->clock());
    clock_to_solver_grp_[m->clock()] = pgrp;
  } else {
    pgrp = grp_iter->second;
  }

  // Use the solver in group first
  Solver<Dtype> *pfc = pgrp->PopSolver();
  if (pfc == NULL) {
    pfc = this->PopFreeSolver();
  }

  // allocate a new solver if failed to find a buffered solver
  if (pfc == NULL) {
    Solver<Dtype> *proot =
                  (Solver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
    const SolverParameter& solver_param = NodeEnv::Instance()->SolverParam();
    pfc = this->NewSolver(proot, solver_param);
  }

  shared_ptr<Net<Dtype> > fc_net = pfc->net();

  CopyInputDataFromMsg(fc_net, m);
  fc_net->ForwardPrefilled();

  this->BindSolver(pfc, m->msg_id());

  // don't need send activation if we are a loss node
  if (this->IsLossNode()) {
    return;
  }

  shared_ptr<Msg> r(new Msg(m));
  // broadcast the message
  r->set_dst(-1);
  if (NodeEnv::Instance()->num_splits() > 1) {
    r->set_is_partial(true);
    r->set_data_offset(NodeEnv::Instance()->node_position());
  }
  ParamHelper<Dtype>::CopyOutputDataToMsg(fc_net, r);

  this->SendMsg(r);
}


template <typename Dtype>
void FcThread<Dtype>::CopyOutputDiffFromMsg(shared_ptr<Net<Dtype> > fc_net,
                                            shared_ptr<Msg> m) {
  for (int i = 0; i < fc_net->num_outputs(); i++) {
    int blob_index = fc_net->output_blob_indices()[i];
    const string& blob_name = fc_net->blob_names()[blob_index];

    Blob<Dtype>* pblob = fc_net->output_blobs()[i];
    ParamHelper<Dtype>::CopyBlobDiffFromMsg(pblob, blob_name, m);
  }
}


template <typename Dtype>
void FcThread<Dtype>::FcBackward(shared_ptr<Msg> m,
                                 bool copy_diff) {
  MLOG(INFO) << "Begin backward for src: " << m->src()
             << ", ID: " << m->conv_id();

  Solver<Dtype> *pfc = this->FindSolver(m->msg_id());
  shared_ptr<Net<Dtype> > fc_net = pfc->net();
  if (copy_diff) {
    CopyOutputDiffFromMsg(fc_net, m);
  }

  fc_net->Backward();

  const vector<int>& pre_ids = NodeEnv::Instance()->prev_node_ids();

  vector<shared_ptr<Msg> > bwd_msgs;
  if (pre_ids.size() <= 0) {  // we are the gateway node
    shared_ptr<Msg> r(new Msg(m));
    r->set_dst(m->src());
    bwd_msgs.push_back(r);
  } else {
    for (int i = 0; i < pre_ids.size(); i++) {
      shared_ptr<Msg> r(new Msg(m));
      r->set_dst(pre_ids[i]);
      bwd_msgs.push_back(r);
    }
  }

  // copy diff to downstream nodes
  for (int i = 0; i < bwd_msgs.size(); i++) {
    shared_ptr<Msg> r = bwd_msgs.at(i);

    r->set_type(BACKWARD);
    ParamHelper<Dtype>::CopyInputDiffToMsg(fc_net, r, i, bwd_msgs.size());
    this->SendMsg(r);
  }

  group_iter_t grp_iter = clock_to_solver_grp_.find(m->clock());
  CHECK(grp_iter != clock_to_solver_grp_.end());

  shared_ptr<SolverGroup<Dtype> > pgrp = grp_iter->second;

  if (this->IsLossNode()) {
    Dtype loss = fc_net->output_blobs()[0]->cpu_data()[0];
    pgrp->AddLoss(loss);
  }
  this->RemoveBind(m->msg_id());
  pgrp->PushSolver(pfc);

  SendGradients(m);
}

template <typename Dtype>
void FcThread<Dtype>::RemoveSolvers(int clock) {
  group_iter_t grp_iter = clock_to_solver_grp_.find(clock);
  if (grp_iter == clock_to_solver_grp_.end()) {
    return;
  }

  shared_ptr<SolverGroup<Dtype> > pgrp = grp_iter->second;

  Solver<Dtype> *p = NULL;
  while ((p = pgrp->PopSolver()) != NULL) {
    this->PushFreeSolver(p);
  }

  clock_to_solver_grp_.erase(grp_iter);
}

template <typename Dtype>
void FcThread<Dtype>::SendGradients(shared_ptr<Msg> m) {
  group_iter_t grp_iter = clock_to_solver_grp_.find(m->clock());
  CHECK(grp_iter != clock_to_solver_grp_.end());

  shared_ptr<SolverGroup<Dtype> > pgrp = grp_iter->second;

  int total_sub_batches = NodeEnv::Instance()->num_sub_solvers()
                                            * this->GetClients();
  if (pgrp->num_sub_batches() < total_sub_batches) {
    return;
  }

  Solver<Dtype> *pfc_grad = pgrp->PopSolver();
  shared_ptr<Net<Dtype> > grad_net = pfc_grad->net();

  Solver<Dtype> *p = NULL;
  while ((p = pgrp->PopSolver()) != NULL) {
    ParamHelper<Dtype>::AddDiffFromNet(grad_net, p->net());
    ParamHelper<Dtype>::ScalDiff(p->net(), (Dtype)0.0);
    this->PushFreeSolver(p);
  }

  if (this->IsLossNode()) {
    Dtype batch_loss = pgrp->total_loss();
    grad_net->output_blobs()[0]->mutable_cpu_data()[0] = batch_loss;
  }
  this->PushFreeSolver(pfc_grad);

  // notify the param thread
  shared_ptr<Msg> notify(new Msg(m));

  notify->set_type(PUT_GRADIENT);
  notify->set_dst(ROOT_THREAD_ID);
  notify->AppendData(&pfc_grad, sizeof(pfc_grad));

  this->SendMsg(notify);
}

template <typename Dtype>
void FcThread<Dtype>::ProcessMsg(shared_ptr<Msg> m) {
  if (m->type() == FORWARD) {
    FcForward(m);
  } else if (m->type() == BACKWARD) {
    FcBackward(m, true);
  } else {
    LOG(INFO) << "unkown type: " << m->type();
  }
}

template <typename Dtype>
void FcThread<Dtype>::Run() {
  #ifdef USE_MKL
  int n = mkl_get_max_threads();
  LOG(INFO) << "max mkl threads: " << n;
  this->BindOMPThreads(this->omp_cores_);
  this->BindCore(this->omp_cores_[0]);
  #endif

  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);
    vector<shared_ptr<Msg> > msgs;

    int clock_bound = clock_ + staleness_;

    if (m->type() == UPDATE_CLOCK) {
      // release the solver associated with clock
      RemoveSolvers(clock_);

      clock_ = m->clock();

      clock_bound = clock_ + staleness_;
      for (int i = 0; i < msg_buf_.size(); i++) {
        if (msg_buf_[i]->clock() <= clock_bound) {
          ProcessMsg(msg_buf_[i]);
        } else {
          msgs.push_back(msg_buf_[i]);
        }
      }
      msg_buf_.clear();

      for (int i = 0; i < msgs.size(); i++) {
        msg_buf_.push_back(msgs[i]);
      }
    } else if (m->type() == EXIT_TRAIN) {
      // exit training
      return;
    } else {
      if (m->clock() <= clock_bound) {
        ProcessMsg(m);
      } else {
        LOG(WARNING) << "Wait for param thread";
        msg_buf_.push_back(m);
      }
    }
  }
}

template <typename Dtype>
boost::atomic_int FcLossThread<Dtype>::iter_(0);

template <typename Dtype>
void FcLossThread<Dtype>::ProcessMsg(shared_ptr<Msg> m) {
  this->FcForward(m);

  this->FcBackward(m, false);

  iter_++;
}

#if 0
template <typename Dtype>
void FcLossThread<Dtype>::Run() {
  #ifdef USE_MKL
  int n = mkl_get_max_threads();
  LOG(INFO) << "max mkl threads: " << n;
  mkl_set_dynamic(false);
  #endif

  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);
  }
}
#endif

template <typename Dtype>
int FcParamThread<Dtype>::GetGroupIndex(void *psolver, int64_t msg_id) {
  int clock = this->GetParamBuf()->GetClock(psolver);
  CHECK_NE(clock, INVALID_CLOCK) << "invalid clock";

  unordered_map<int, int>::iterator iter = clock_to_group_idx_.find(clock);
  if (iter != clock_to_group_idx_.end()) {
    return iter->second;
  }

  // find or allocate a new slot the store the group solver
  int i = 0;
  for (i = 0; i < group_solvers_.size(); i++) {
    if (group_solvers_[i] == NULL) {
      break;
    }
  }

  if (i >= group_solvers_.size()) {
    group_solvers_.push_back(psolver);
    grad_updates_vec_.push_back(0);
    msg_id_vec_.push_back(msg_id);
    group_loss_vec_.push_back(0);
    clock_vec_.push_back(clock);
    clock_to_group_idx_[clock] = group_solvers_.size() - 1;
  } else {
    group_solvers_[i] = psolver;
    grad_updates_vec_[i] = 0;
    msg_id_vec_[i] = msg_id;
    group_loss_vec_[i] = 0;
    clock_vec_[i] = clock;
    clock_to_group_idx_[clock] = i;
  }

  return i;
}

template <typename Dtype>
void FcParamThread<Dtype>::ClearGroup(int grp_idx) {
  Solver<Dtype> *pgroup_solver = (Solver<Dtype> *)group_solvers_[grp_idx];
  CHECK(pgroup_solver != NULL);

  ParamHelper<Dtype>::ScalDiff(pgroup_solver->net(), (Dtype)0.0);

  this->GetParamBuf()->RemoveClock(clock_vec_[grp_idx]);
  this->GetParamBuf()->DeRefParam(pgroup_solver);

  unordered_map<int, int>::iterator iter =
                                clock_to_group_idx_.find(clock_vec_[grp_idx]);
  clock_to_group_idx_.erase(iter);

  group_solvers_[grp_idx] = NULL;
  group_loss_vec_[grp_idx] = 0;
  grad_updates_vec_[grp_idx] = 0;
  msg_id_vec_[grp_idx] = INVALID_NODE_ID;
  clock_vec_[grp_idx] = INVALID_CLOCK;
}

template <typename Dtype>
void FcParamThread<Dtype>::AddGradients(shared_ptr<Msg> m) {
  Solver<Dtype> *psolver = ((Solver<Dtype> **)m->ZmsgData(0))[0];
  CHECK(psolver != NULL);

  SGDSolver<Dtype> *proot =
                  (SGDSolver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
  const vector<Blob<Dtype>*>& root_output = proot->net()->output_blobs();

  if (this->IsLossNode()) {
    const vector<Blob<Dtype>*>& output = psolver->net()->output_blobs();
    CHECK_EQ(output.size(), 1) << "only deal with output size 1";
    Blob<Dtype>* pblob = output[0];
    root_output[0]->mutable_cpu_data()[0] += pblob->cpu_data()[0];
  }

  ParamHelper<Dtype>::AddDiffFromNet(proot->net(), psolver->net());
  // clear diff params
  ParamHelper<Dtype>::ScalDiff(psolver->net(), (Dtype)0.0);
}


template <typename Dtype>
int FcParamThread<Dtype>::ProcessGradients(shared_ptr<Msg> m) {
  grad_msgs_.push_back(m);

  int batch_updates = std::min(fc_threads_, num_conv_workers_);
  if (grad_msgs_.size() < batch_updates) {
    return 0;
  }

  for (int i = 0; i < grad_msgs_.size(); i++) {
    AddGradients(grad_msgs_[i]);
  }
  grad_msgs_.clear();

  return UpdateParam();
}

template <typename Dtype>
int FcParamThread<Dtype>::UpdateParam() {
  SGDSolver<Dtype> *proot =
                  (SGDSolver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
  const vector<Blob<Dtype>*>& root_output = proot->net()->output_blobs();

  Dtype s = (Dtype)(1.0 / (Dtype)(num_conv_workers_ * this->num_sub_solvers_));

  if (this->IsLossNode()) {
    Dtype loss = root_output[0]->cpu_data()[0] * s;
    LOG(INFO) << "train iteration: " << train_iter_
              << " loss: " << loss;
    root_output[0]->mutable_cpu_data()[0] = 0;
  }

  // scaling gradients
  ParamHelper<Dtype>::ScalDiff(proot->net(), s);

  proot->CommitGradient();
  this->UpdateSocketParams();

  ParamHelper<Dtype>::ScalDiff(proot->net(), (Dtype)0.0);

  UpdateClock();

  if (train_iter_ == max_iter_) {
    StopModelServer();
  }

  if (test_node_id_ > 0
     && (train_iter_ % TRAIN_NOTIFY_INTERVAL == 0
        || train_iter_ >= max_iter_)
     && this->IsLossNode()
     ) {
    this->SendNotify();
  }

  if (test_node_id_ < 0 && train_iter_ >= max_iter_) {
    return -1;
  }

  return 0;
}

template <typename Dtype>
void FcParamThread<Dtype>::StopModelServer() {
  string ms_addr(NodeEnv::model_server_addr());
  int node_id = NodeEnv::Instance()->ID();

  shared_ptr<SkSock> dealer(new SkSock(ZMQ_DEALER));
  dealer->SetId(node_id);
  dealer->Connect(ms_addr);

  shared_ptr<Msg> m(new Msg());
  m->set_type(EXIT_TRAIN);
  m->set_src(node_id);
  m->set_clock(train_iter_);

  int pad = 0;
  m->AppendData(&pad, sizeof(pad));

  dealer->SendMsg(m);

  // notify id server
  shared_ptr<SkSock> req(new SkSock(ZMQ_REQ));
  string id_addr(NodeEnv::id_server_addr());
  req->Connect(id_addr);

  req->SendMsg(m);
}

template <typename Dtype>
void FcParamThread<Dtype>::UpdateClock() {
  train_iter_++;

  shared_ptr<Msg> r(new Msg());
  r->set_type(UPDATE_CLOCK);
  r->set_dst(WORKER_BCAST);
  r->set_src(NodeEnv::Instance()->ID());
  r->set_clock(train_iter_);

  r->AppendData(&train_iter_, sizeof(train_iter_));

  this->SendMsg(r);
}

template <typename Dtype>
void FcParamThread<Dtype>::SendNotify() {
  shared_ptr<Msg> r(new Msg());
  r->set_type(TRAIN_ITER);
  r->set_dst(test_node_id_);
  r->set_src(NodeEnv::Instance()->ID());

  r->AppendData(&train_iter_, sizeof(train_iter_));

  // LOG(INFO) << "sending notify";

  this->SendMsg(r);
}


template <typename Dtype>
int FcParamThread<Dtype>::SendParam(shared_ptr<Msg> m) {
  shared_ptr<Msg> r(new Msg());
  r->set_type(PUT_PARAM);
  r->set_dst(m->src());
  r->set_src(NodeEnv::Instance()->ID());

  Solver<Dtype> *psolver =
                (Solver<Dtype> *)NodeEnv::Instance()->GetRootSolver();

  shared_ptr<Net<Dtype> > net = psolver->net();
  ParamHelper<Dtype>::CopyParamDataToMsg(net, net->layer_names(), r);

  // LOG(INFO) << "sending param";

  this->SendMsg(r);

  if (train_iter_ > max_iter_) {
    return -1;
  }

  return 0;
}

template <typename Dtype>
void FcParamThread<Dtype>::Run() {
  #ifdef USE_MKL
  int fc_omp_threads = mkl_get_max_threads();
  mkl_set_num_threads_local(fc_omp_threads * fc_threads_);

  int n = mkl_get_max_threads();
  LOG(INFO) << "max mkl threads in param thread: " << n;

  this->BindOMPThreads(this->omp_cores_);

  int last_core = NodeEnv::Instance()->GetOnlineCores() - 1;
  this->BindCore(last_core);
  #endif

  // use the root solver
  Caffe::set_root_solver(true);

  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);

    if (m->type() == GET_PARAM) {
      test_node_id_ = m->src();
      if (SendParam(m) < 0) {
        this->SendExit();
        return;
      }
    } else if (m->type() == EXIT_TRAIN) {
      // exit training
      this->SendExit();
      return;
    } else if (m->type() == PUT_GRADIENT) {
      if (ProcessGradients(m) < 0) {
        this->SendExit();
        return;
      }
    } else {
      LOG(ERROR) << "Unknown type: " << m->type();
    }
  }
}

INSTANTIATE_CLASS(ParamBuf);
INSTANTIATE_CLASS(SolverGroup);
INSTANTIATE_CLASS(FcWorker);
INSTANTIATE_CLASS(FcThread);
INSTANTIATE_CLASS(FcLossThread);
INSTANTIATE_CLASS(FcParamThread);


}  // end namespace caffe


