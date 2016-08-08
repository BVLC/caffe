
#include "caffe/multi_node/fc_thread.hpp"
#include "caffe/multi_node/param_helper.hpp"

namespace caffe {

template <typename Dtype>
ParamBuf<Dtype> *FcWorker<Dtype>::pbuf_ = NULL;

template <typename Dtype>
boost::once_flag FcWorker<Dtype>::once_;

template <typename Dtype>
vector<Blob<Dtype>*> * ParamBuf<Dtype>::RefParam(void *psolver, int clock)
{
  boost::mutex::scoped_lock rlock(ref_mutex_);
  
  psolver_to_clock_[psolver] = clock;
  unordered_map<int, int>::iterator clock_iter = clock_to_idx_.find(clock);

  int idx = -1;
  if (clock_iter != clock_to_idx_.end()) {
    // use the param associated with the clock
    idx = clock_iter->second;
  } else {
    // use the latest param for this clock
    unordered_map<void *, int>::iterator iter = pointer_to_idx_.find(platest_param_);
    CHECK(iter != pointer_to_idx_.end()) << "cannot find index to pointer: " << platest_param_;
    idx = iter->second;
    clock_to_idx_[clock] = idx;
  }
  
  psolver_to_idx_[psolver] = idx;
  ref_cnt_vec_[idx]++;

  return param_vec_[idx];
}

template <typename Dtype>
vector<Blob<Dtype>*> *ParamBuf<Dtype>::FindParam(void *psolver)
{
  boost::mutex::scoped_lock lock(ref_mutex_);
  
  unordered_map<void *, int>::iterator iter = psolver_to_idx_.find(psolver);
  CHECK(iter != psolver_to_idx_.end()) << "cannot find index to pointer: " << psolver;

  int idx = iter->second;

  return param_vec_[idx];
}


template <typename Dtype>
int ParamBuf<Dtype>::DeRefParam(void *psolver)
{
  boost::mutex::scoped_lock lock(ref_mutex_);
  
  unordered_map<void *, int>::iterator iter = psolver_to_idx_.find(psolver);
  CHECK(iter != psolver_to_idx_.end()) << "cannot find index to pointer: " << psolver;

  int idx = iter->second;
  
  psolver_to_idx_.erase(iter);
  ref_cnt_vec_[idx]--;
  CHECK_GE(ref_cnt_vec_[idx], 0) << "unexpected reference counter";

  unordered_map<void *, int>::iterator clock_iter = psolver_to_clock_.find(psolver);
  CHECK(clock_iter != psolver_to_clock_.end());

  psolver_to_clock_.erase(clock_iter);

  return ref_cnt_vec_[idx];
}


template <typename Dtype>
vector<Blob<Dtype>*> *ParamBuf<Dtype>::CreateParam(const vector<Blob<Dtype>*> &params)
{
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
void ParamBuf<Dtype>::InitParamBuf(const vector<Blob<Dtype>*> &params)
{
  // create 4 paramters in the beginning
  for (int i = 0; i < 4; i++) {
    CreateParam(params);
  }
}

template <typename Dtype>
vector<Blob<Dtype>*> *ParamBuf<Dtype>::GetParam()
{
  return platest_param_;
}

template <typename Dtype>
vector<Blob<Dtype>*> *ParamBuf<Dtype>::FindFreeParam()
{
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
void ParamBuf<Dtype>::ReplaceParam(vector<Blob<Dtype>*> *p)
{
  boost::mutex::scoped_lock lock(ref_mutex_);
  platest_param_ = p;
}

template <typename Dtype>
void FcThread<Dtype>::CopyInputDataFromMsg(shared_ptr<Net<Dtype> > fc_net, shared_ptr<Msg> m)
{
  for (int i = 0; i < fc_net->num_inputs(); i++) {
    int blob_index = fc_net->input_blob_indices()[i];
    const string& blob_name = fc_net->blob_names()[blob_index];

    Blob<Dtype>* pblob = fc_net->input_blobs()[i];
    ParamHelper<Dtype>::CopyBlobDataFromMsg(pblob, blob_name, m);
  }
}


template <typename Dtype>
shared_ptr<Msg> FcThread<Dtype>::FcForward(shared_ptr<Msg> m)
{
  Solver<Dtype> *pfc = (Solver<Dtype> *)NodeEnv::Instance()->PopFreeSolver();
  Solver<Dtype> *proot = (Solver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
  
  if (NULL == pfc) {
    const SolverParameter& solver_param = NodeEnv::Instance()->SolverParam();

    pfc = (Solver<Dtype> *)this->NewSolver(proot, solver_param);
  }

  // copy param data from root solver
  const vector<Blob<Dtype>*>& params = pfc->net()->learnable_params();
  // const vector<Blob<Dtype>*>& root_params = proot->net()->learnable_params();

  const vector<Blob<Dtype>*> *ref_params = this->GetParamBuf()->RefParam(pfc, m->clock());
  CHECK_EQ(params.size(), ref_params->size());
  
  for (int i = 0; i < params.size(); i++) {
    // CHECK_EQ(params[i]->count(), ref_params->at(i)->count());
    // ParamHelper<Dtype>::BlasCopy(root_params[i]->count(), root_params[i]->cpu_data(), params[i]->mutable_cpu_data());
    params[i]->ShareData(*ref_params->at(i));
  }

  // ParamHelper<Dtype>::PrintParam(pfc->net());
  
  shared_ptr<Net<Dtype> > fc_net = pfc->net();
  
  CopyInputDataFromMsg(fc_net, m);
  fc_net->ForwardPrefilled();
  
  shared_ptr<Msg> r(new Msg(m));
  // broadcast the message
  r->set_dst(-1);
  if (NodeEnv::Instance()->num_splits() > 1) {
    r->set_is_partial(true);
    r->set_data_offset(NodeEnv::Instance()->node_position());
  }
  ParamHelper<Dtype>::CopyOutputDataToMsg(fc_net, r);

  NodeEnv::Instance()->PutSolver(m->msg_id(), pfc);

  return r;
}


template <typename Dtype>
void FcThread<Dtype>::CopyOutputDiffFromMsg(shared_ptr<Net<Dtype> > fc_net, shared_ptr<Msg> m)
{
  for (int i = 0; i < fc_net->num_outputs(); i++) {
    int blob_index = fc_net->output_blob_indices()[i];
    const string& blob_name = fc_net->blob_names()[blob_index];
    
    Blob<Dtype>* pblob = fc_net->output_blobs()[i];
    ParamHelper<Dtype>::CopyBlobDiffFromMsg(pblob, blob_name, m);
  }
}


template <typename Dtype>
void FcThread<Dtype>::FcBackward(shared_ptr<Msg> m, vector<shared_ptr<Msg> >& replies, bool copy_diff)
{
  Solver<Dtype> *pfc = (Solver<Dtype> *)NodeEnv::Instance()->FindSolver(m->msg_id());
  CHECK(pfc != NULL);
  
  shared_ptr<Net<Dtype> > fc_net = pfc->net();
  if (copy_diff) {
    CopyOutputDiffFromMsg(fc_net, m);
  }

  fc_net->Backward();

  const vector<int>& pre_ids = NodeEnv::Instance()->prev_node_ids();

  if (pre_ids.size() <= 0) { // we are the gateway node
    shared_ptr<Msg> r(new Msg(m));
    r->set_dst(m->src());
    replies.push_back(r);
  } else {
    for (int i = 0; i < pre_ids.size(); i++) {
      shared_ptr<Msg> r(new Msg(m));
      r->set_dst(pre_ids[i]);
      replies.push_back(r);
    }
  }
  
  //copy diff to downstream nodes
  for (int i = 0; i < replies.size(); i++) {
    shared_ptr<Msg> r = replies[i];

    r->set_type(BACKWARD);
    ParamHelper<Dtype>::CopyInputDiffToMsg(fc_net, r, i, replies.size());
  }
  
  // pfc->UpdateDiff();
  
  //notify the param thread
  shared_ptr<Msg> notify(new Msg(m));

  notify->set_dst(ROOT_THREAD_ID);
  notify->AppendData(&pfc, sizeof(pfc));
  replies.push_back(notify);
}

template <typename Dtype>
void FcThread<Dtype>::ProcessMsg(shared_ptr<Msg> m)
{
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

template <typename Dtype>
void FcThread<Dtype>::Run()
{
  #ifdef USE_MKL
  int n = mkl_get_max_threads();
  LOG(INFO) << "max mkl threads: " << n;
  mkl_set_dynamic(false);
  #endif

  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);
    vector<shared_ptr<Msg> > msgs;
    
    int clock_bound = clock_ + staleness_;

    if (m->type() == UPDATE_CLOCK) {
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
    } else {
      if (m->clock() <= clock_bound) {
        ProcessMsg(m);
      } else {
        msg_buf_.push_back(m);
      }
    }
  }
}

template <typename Dtype>
boost::atomic_int FcLossThread<Dtype>::iter_(0);

template <typename Dtype>
void FcLossThread<Dtype>::ProcessMsg(shared_ptr<Msg> m)
{
  shared_ptr<Msg> f = this->FcForward(m);
  
  vector<shared_ptr<Msg> > replies;
  this->FcBackward(f, replies, false);

  iter_++;

  for (int i = 0; i < replies.size(); i++) {
    this->SendMsg(replies[i]);
  }
}

#if 0
template <typename Dtype>
void FcLossThread<Dtype>::Run()
{
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
void FcParamThread<Dtype>::ClearGroup(int grp_idx)
{
  Solver<Dtype> *pgroup_solver = (Solver<Dtype> *)group_solvers_[grp_idx];
  CHECK(pgroup_solver != NULL);

  ParamHelper<Dtype>::ScalDiff(pgroup_solver->net(), (Dtype)0.0);
  
  this->GetParamBuf()->RemoveClock(clock_vec_[grp_idx]);
  this->GetParamBuf()->DeRefParam(pgroup_solver);

  NodeEnv::Instance()->DeleteSolver(msg_id_vec_[grp_idx]);
  NodeEnv::Instance()->PushFreeSolver(pgroup_solver);
  
  unordered_map<int, int>::iterator iter = clock_to_group_idx_.find(clock_vec_[grp_idx]);
  clock_to_group_idx_.erase(iter);
  
  group_solvers_[grp_idx] = NULL;
  group_loss_vec_[grp_idx] = 0;
  grad_updates_vec_[grp_idx] = 0;
  msg_id_vec_[grp_idx] = INVALID_ID;
  clock_vec_[grp_idx] = INVALID_CLOCK;
}

template <typename Dtype>
void FcParamThread<Dtype>::UpdateParam(shared_ptr<Msg> m)
{
  Solver<Dtype> *psolver = (Solver<Dtype> *)NodeEnv::Instance()->FindSolver(m->msg_id());
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
  

  #if 0
  if (sub_batches_ == 0) {
    ParamHelper<Dtype>::CopyDiffFromNet(proot->net(), psolver->net());
  } else {
    ParamHelper<Dtype>::AddDiffFromNet(proot->net(), psolver->net());
  }
  
  // doesn't have broadcast nodes means we are loss layer
  const vector<string>& bcast_addrs = NodeEnv::Instance()->bcast_addrs();

  if (bcast_addrs.size() == 0) {
    const vector<Blob<Dtype>*>& output = psolver->net()->output_blobs();
    CHECK_EQ(output.size(), 1) << "only deal with output size 1";
    Blob<Dtype>* pblob = output[0];
    sub_loss_ += pblob->cpu_data()[0];
  }
 
  // clear diff params
  ParamHelper<Dtype>::ScalDiff(psolver->net(), (Dtype)0.0);
  
  this->GetParamBuf()->DeRefParam(psolver);
  
  NodeEnv::Instance()->DeleteSolver(m->msg_id());
  NodeEnv::Instance()->PushFreeSolver(psolver);
 
  sub_batches_++;
  if (sub_batches_ < num_workers_ * NUM_SUB_SOLVERS) {
    return;
  }
  #endif
  
  int group_id = GetGroupIndex(psolver, m->msg_id());

  Solver<Dtype> *pgroup_solver = (Solver<Dtype> *)group_solvers_[group_id];
  // doesn't have broadcast nodes means we are loss layer
  const vector<string>& bcast_addrs = NodeEnv::Instance()->bcast_addrs();
 
  if (bcast_addrs.size() == 0) {
    const vector<Blob<Dtype>*>& output = psolver->net()->output_blobs();
    CHECK_EQ(output.size(), 1) << "only deal with output size 1";
    Blob<Dtype>* pblob = output[0];
    group_loss_vec_[group_id] += pblob->cpu_data()[0];
  }
  grad_updates_vec_[group_id]++;

  if (pgroup_solver != psolver) {
    ParamHelper<Dtype>::AddDiffFromNet(pgroup_solver->net(), psolver->net());
    
    // clear diff params
    ParamHelper<Dtype>::ScalDiff(psolver->net(), (Dtype)0.0);
  
    this->GetParamBuf()->DeRefParam(psolver);
  
    NodeEnv::Instance()->DeleteSolver(m->msg_id());
    NodeEnv::Instance()->PushFreeSolver(psolver);
    // LOG(INFO) << "release solver for group id: " << group_id;
  } else {
    // LOG(INFO) << "keep solver for group id: " << group_id;
  }
  
  if (grad_updates_vec_[group_id] < num_workers_ * NUM_SUB_SOLVERS) {
    return;
  }

  // share paramters
  const vector<Blob<Dtype>*>& root_params = proot->net()->learnable_params();
  vector<Blob<Dtype>*> *param = this->GetParamBuf()->FindFreeParam();
  if (param == NULL) {
    param = this->GetParamBuf()->CreateParam(root_params);
  }

  // 
  CHECK_EQ(root_params.size(), param->size());
  
  for (int i = 0; i < root_params.size(); i++) {
    CHECK_EQ(root_params[i]->count(), param->at(i)->count());
    
    if (root_params[i]->cpu_data() != param->at(i)->cpu_data()) {
      ParamHelper<Dtype>::BlasCopy(root_params[i]->count(), root_params[i]->cpu_data(), param->at(i)->mutable_cpu_data());
      root_params[i]->ShareData(*param->at(i));
    }
  }

  ParamHelper<Dtype>::CopyDiffFromNet(proot->net(), pgroup_solver->net());
  
  // scaling gradients
  Dtype s = (Dtype)(1.0 / (Dtype)(num_workers_ * NUM_SUB_SOLVERS));
  ParamHelper<Dtype>::ScalDiff(proot->net(), s);
  
  proot->CommitGradient();
  
  if (bcast_addrs.size() == 0) {
    group_loss_vec_[group_id] *= s;
    LOG(INFO) << "train iteration: " << train_iter_ << " loss: " << group_loss_vec_[group_id];
  }
 
  ClearGroup(group_id);

  // switch the working param
  this->GetParamBuf()->ReplaceParam(param);
  
  #if 0
  ParamHelper<Dtype>::PrintParam(proot->net());
  ParamHelper<Dtype>::PrintDiff(proot->net());
  #endif
  
  UpdateClock();

  if (test_node_id_ > 0  
     && train_iter_ % TRAIN_NOTIFY_INTERVAL == 0
     &&  bcast_addrs.size() == 0
      ) {
    this->SendNotify();
  }
}

template <typename Dtype>
void FcParamThread<Dtype>::UpdateClock()
{
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
  #ifdef USE_MKL
  if (this->omp_threads_ > 0) {
    mkl_set_num_threads_local(this->omp_threads_);
  }
  int n = mkl_get_max_threads();
  LOG(INFO) << "max mkl threads in param thread: " << n;
  mkl_set_dynamic(false);
  #endif

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

INSTANTIATE_CLASS(ParamBuf);
INSTANTIATE_CLASS(FcWorker);
INSTANTIATE_CLASS(FcThread);
INSTANTIATE_CLASS(FcLossThread);
INSTANTIATE_CLASS(FcParamThread);


} //end caffe


