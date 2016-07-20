
#include "caffe/multi_node/conv_thread.hpp"
#include "caffe/multi_node/param_helper.hpp"

namespace caffe {

template <typename Dtype>
boost::mutex ConvThread<Dtype>::conv_id_mutex_;

template <typename Dtype>
int ConvThread<Dtype>::conv_id_ = 1;

template <typename Dtype>
boost::barrier *ConvThread<Dtype>::pconv_barrier_ = NULL;

template <typename Dtype>
SGDSolver<Dtype> *ConvThread<Dtype>::full_solver_ = NULL;


template <typename Dtype>
void ConvThread<Dtype>::ForwardLayer(shared_ptr<Net<Dtype> > conv_net, int layer_id)
{
  shared_ptr<Layer<Dtype> > l = conv_net->layers()[layer_id];
  const vector<Blob<Dtype>*>& bottom = conv_net->bottom_vecs()[layer_id];
  const vector<Blob<Dtype>*>& top = conv_net->top_vecs()[layer_id];
  
  string skip_layer("LRN");
  
  if (skip_layer == l->type()) {
    shared_ptr<Net<Dtype> > full_net = full_solver_->net();
    const vector<Blob<Dtype>*>& full_bottom = full_net->bottom_vecs()[layer_id];
    const vector<Blob<Dtype>*>& full_top = full_net->top_vecs()[layer_id];

    // copy bottom blobs to full solver
    ParamHelper<Dtype>::CopyBlobData(bottom, full_bottom, this->GetWorkerId());
    
    pconv_barrier_->wait();
    if (this->GetWorkerId() == 0) {
      #ifdef USE_MKL
      int orig_thread = mkl_set_num_threads_local(this->GetWorkerNum());
      #endif
      
      shared_ptr<Layer<Dtype> > full_layer = full_net->layers()[layer_id];
      full_layer->Forward(full_bottom, full_top);
  
      #ifdef USE_MKL
      mkl_set_num_threads_local(orig_thread);
      #endif
    }
    
    pconv_barrier_->wait();
    ParamHelper<Dtype>::CopyBlobData(full_top, top, this->GetWorkerId());
  } else {
    l->Forward(bottom, top);
  }
}

template <typename Dtype>
void ConvThread<Dtype>::BackwardLayer(shared_ptr<Net<Dtype> > conv_net, int layer_id)
{
  shared_ptr<Layer<Dtype> > l = conv_net->layers()[layer_id];
  const vector<Blob<Dtype>*>& bottom = conv_net->bottom_vecs()[layer_id];
  const vector<Blob<Dtype>*>& top = conv_net->top_vecs()[layer_id];
  
  string skip_layer("LRN");

  const vector<bool>& layer_need_bwd = conv_net->layer_need_backward();
  if (!layer_need_bwd[layer_id]) {
    return;
  }

  const vector<vector<bool> >& bottom_need_bwd = conv_net->bottom_need_backward();
  
  if (skip_layer == l->type()) {
    shared_ptr<Net<Dtype> > full_net = full_solver_->net();
    const vector<Blob<Dtype>*>& full_bottom = full_net->bottom_vecs()[layer_id];
    const vector<Blob<Dtype>*>& full_top = full_net->top_vecs()[layer_id];
    
    ParamHelper<Dtype>::CopyBlobDiff(top, full_top, this->GetWorkerId());

    pconv_barrier_->wait();
    if (this->GetWorkerId() == 0) {
      shared_ptr<Layer<Dtype> > full_layer = full_net->layers()[layer_id];
      full_layer->Backward(full_top, bottom_need_bwd[layer_id], full_bottom);
    }

    pconv_barrier_->wait();
    ParamHelper<Dtype>::CopyBlobDiff(full_bottom, bottom, this->GetWorkerId());
  } else {
    l->Backward(top, bottom_need_bwd[layer_id], bottom);
  }
}

template <typename Dtype>
void ConvThread<Dtype>::ConvForward()
{
  WorkerSolver<Dtype> *pconv = (WorkerSolver<Dtype> *)NodeEnv::Instance()->PopFreeSolver();
  if (NULL == pconv) {
    Solver<Dtype> *root_solver = (Solver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
    const SolverParameter& solver_param = NodeEnv::Instance()->SolverParam();
    pconv = (WorkerSolver<Dtype> *)this->NewSolver(root_solver, solver_param);
  }

  shared_ptr<Net<Dtype> > conv_net = pconv->net();
  conv_net->ClearParamDiffs();
  conv_net->ForwardPrefilled();
  
  /*
  for (int i = 0; i < conv_net->layers().size(); i++) {
    ForwardLayer(conv_net, i);
  }
  */
  
  // notify the param thread
  shared_ptr<Msg> m(new Msg());
  int conv_id = NewConvId();
  m->set_conv_id(conv_id);
  m->set_src(this->GetWorkerId());
  m->set_type(FORWARD);
  m->set_dst(ROOT_THREAD_ID);
  
  // append the solver pointer
  m->AppendData(&pconv, sizeof(pconv));
  
  this->SendMsg(m);

  NodeEnv::Instance()->PutSolver(conv_id, pconv);
}


template <typename Dtype>
WorkerSolver<Dtype> *ConvThread<Dtype>::PrepareBackwardSolver(shared_ptr<Msg> m)
{
  int conv_id = m->conv_id();
  WorkerSolver<Dtype> *pconv = (WorkerSolver<Dtype> *)NodeEnv::Instance()->FindSolver(conv_id);
  CHECK(pconv != NULL) << "cannot find conv_id: " << conv_id;
  
  /*
  shared_ptr<Net<Dtype> > conv_net = pconv->net();
  ParamHelper<Dtype>::CopyOutputDiffFromMsg(conv_net, m);
  */

  return pconv;
}

template <typename Dtype>
void ConvThread<Dtype>::SyncLayer(int conv_id,int layer_id)
{
  shared_ptr<Msg> m(new Msg());
  m->set_src(this->GetWorkerId());
  m->set_dst(ROOT_THREAD_ID);
  m->set_type(PUT_GRADIENT);
  m->set_conv_id(conv_id);

  m->AppendData(&layer_id, sizeof(layer_id));

  this->SendMsg(m);
}

template <typename Dtype>
bool ConvThread<Dtype>::SyncedBackward(Solver<Dtype> *prev_solver, int prev_idx, shared_ptr<Msg> m)
{
  int conv_id = m->conv_id();
  WorkerSolver<Dtype> * pconv = PrepareBackwardSolver(m);
  shared_ptr<Net<Dtype> > conv_net = pconv->net();
  const vector<shared_ptr<Layer<Dtype> > >& layers = conv_net->layers();
  
  shared_ptr<Net<Dtype> > prev_net;
  if (prev_solver != NULL) {
    prev_net = prev_solver->net();
  }

  // backward and sync per layer
  for (int i = layers.size() - 1; i >= prev_idx; i--) {
    conv_net->BackwardFromTo(i, i);
    // BackwardLayer(conv_net, i);

    if (prev_net != NULL) {
      ParamHelper<Dtype>::AddDiffFromNet(conv_net, prev_net, i);
      ParamHelper<Dtype>::ScalDiff(conv_net, (Dtype)(1.0 / NUM_SUB_SOLVERS), i);
    }

    SyncLayer(conv_id, i);
  }

  for (int i = prev_idx - 1; i >= 0; i--) {
    conv_net->BackwardFromTo(i, i);
    // BackwardLayer(conv_net, i);
    
    if (prev_net != NULL) {
      prev_net->BackwardFromTo(i, i);
      ParamHelper<Dtype>::AddDiffFromNet(conv_net, prev_net, i);
      ParamHelper<Dtype>::ScalDiff(conv_net, (Dtype)(1.0 / NUM_SUB_SOLVERS), i);
      // BackwardLayer(prev_net, i);
    }

    SyncLayer(conv_id, i);
  }

  return true;
}


template <typename Dtype>
bool ConvThread<Dtype>::ConvBackward(WorkerSolver<Dtype> *pconv, WorkerSolver<Dtype> *prev_conv, bool peek_next)
{
  shared_ptr<Net<Dtype> > conv_net = pconv->net();

  const vector<shared_ptr<Layer<Dtype> > >& layers = conv_net->layers();
  
  // do backward from layer to layer
  // and sync with PS when a new backward msg becomes ready
  for (int i = layers.size() - 1; i >= 0; i--) {
    // check whether we can do full backwards
    shared_ptr<Msg> r;
    if (peek_next) {
      if ((r = this->RecvMsg(false)) != NULL) {
        return SyncedBackward(pconv, i, r);
      }
    } else {
      conv_net->BackwardFromTo(i, i);
      // BackwardLayer(conv_net, i);
      if (prev_conv != NULL) {
        ParamHelper<Dtype>::AddDiffFromNet(conv_net, prev_conv->net(), i);
      }
    }
  }

  return false;
}


template <typename Dtype>
void ConvThread<Dtype>::Run()
{
  #ifdef USE_MKL
  int n = mkl_get_max_threads();
  LOG(INFO) << "max mkl threads: " << n;
  mkl_set_dynamic(false);
  #endif

  while (!this->must_stop()) {
    for (int i = 0; i < NUM_SUB_SOLVERS; i++) {
      ConvForward();
    }

    bool is_done = false;
    WorkerSolver<Dtype> *prev_solver = NULL;
    int prev_conv_id = -1;

    for (int i = 0; i < NUM_SUB_SOLVERS - 1; i++) {
      shared_ptr<Msg> r;
      
      if ( (r = this->RecvMsg(true)) != NULL) {  //blocked
        WorkerSolver<Dtype> *psolver = PrepareBackwardSolver(r);
        
        bool peek_next = false;
        if (i == (NUM_SUB_SOLVERS - 2)) {
          peek_next = true;
        }

        is_done = ConvBackward(psolver, prev_solver, peek_next);

        if (prev_solver != NULL) {
          NodeEnv::Instance()->DeleteSolver(prev_conv_id);
          NodeEnv::Instance()->PushFreeSolver(prev_solver);
        }

        prev_conv_id = r->conv_id();
        prev_solver = psolver;
      }
    }
    
    if (!is_done) {
      shared_ptr<Msg> r = this->RecvMsg(true);
      SyncedBackward(prev_solver, 0, r);
    }
    
    // delete the conv_solver
    if (prev_solver != NULL) {
      NodeEnv::Instance()->DeleteSolver(prev_conv_id);
      NodeEnv::Instance()->PushFreeSolver(prev_solver);
    }

    // waiting for parameter update
    shared_ptr<Msg> m = this->RecvMsg(true);
    while (m->type() != PUT_PARAM) {
      m = this->RecvMsg(true);
    }
  }
}

template <typename Dtype>
void ConvParamThread<Dtype>::SyncLayer(int layer_id)
{
  SGDSolver<Dtype> *root_solver = (SGDSolver<Dtype> *) NodeEnv::Instance()->GetRootSolver();
  shared_ptr<Net<Dtype> > root_net = root_solver->net();
  const string& layer_name = root_net->layer_names()[layer_id];
  
  layer_update_map_[layer_name] = true;

  // check whether the PS can be updated
  map<string, int>::iterator iter = layer_to_ps_id_.find(layer_name);
  
  // there are split layers in caffe which are not in the layer database
  if (iter == layer_to_ps_id_.end()) {
    return;
  }

  int ps_id = iter->second;
  const vector<string>& ps_layers = NodeEnv::Instance()->FindPSLayer(ps_id);
  for (int i = 0; i < ps_layers.size(); i++) {
    if (!layer_update_map_[ps_layers[i]]) {
      return;
    }
  }
  
  // Put gradient to PS
  shared_ptr<Msg> ps_msg(new Msg());
  ps_msg->set_type(PUT_GRADIENT);
  ps_msg->set_dst(ps_id);
  ps_msg->set_src(NodeEnv::Instance()->ID());
  
  int ps_local_index = ps_id_map_[ps_id];
  ps_clocks_[ps_local_index]++;
  ps_msg->set_clock(ps_clocks_[ps_local_index]);
  
  ParamHelper<Dtype>::CopyParamDiffToMsg(root_net, ps_layers, ps_msg);
  
  this->SendMsg(ps_msg);

  // reset the map
  for (int i = 0; i < ps_layers.size(); i++) {
    layer_update_map_[ps_layers[i]] = false;
  }
}

template <typename Dtype>
void ConvParamThread<Dtype>::SyncWithPS()
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
}


template <typename Dtype>
void ConvParamThread<Dtype>::SendActivations()
{
  vector<shared_ptr<Net<Dtype> > > net_vec;

  // prepare the forward nets from conv_threads
  for (int i = 0; i < fwd_msgs_.size(); i++) {
    SGDSolver<Dtype> *pconv = ((SGDSolver<Dtype> **)fwd_msgs_[i]->ZmsgData(0))[0];
    net_vec.push_back(pconv->net());
  }

  shared_ptr<Msg> m(new Msg());

  ParamHelper<Dtype>::CopyOutputDataToMsg(net_vec, m);
  
  // m is a template message for forwarding
  m->set_src(NodeEnv::Instance()->ID());
  int conv_id = fwd_msgs_[0]->conv_id();
  m->set_type(FORWARD);
  m->set_conv_id(conv_id);

  // always use the 0th clock
  m->set_clock(ps_clocks_[0]);
  m->set_dst(-1);
  
  for (int i = 0; i < gateway_ids_.size(); i++) {
    shared_ptr<Msg> f(new Msg(m));
    f->set_dst(gateway_ids_[i]);
    ParamHelper<Dtype>::CopyBlobDataToMsg(net_vec, gateway_blobs_[i], f);
    this->SendMsg(f);
  }
  
  for (int i = 0; i < fwd_blobs_.size(); i++) {
    shared_ptr<Msg> f(new Msg(m));
    f->set_dst(fwd_ids_[i]);
    ParamHelper<Dtype>::CopyBlobDataToMsg(net_vec, fwd_blobs_[i], f);
    this->SendMsg(f);
  }

  // backup the messages to vector
  shared_ptr<vector<shared_ptr<Msg> > > pvec(new vector<shared_ptr<Msg> >(fwd_msgs_));
  conv_id_to_vec_[conv_id] = pvec;
}

template <typename Dtype>
void ConvParamThread<Dtype>::ProcessForward(shared_ptr<Msg> m)
{
  fwd_msgs_.push_back(m);

  if (fwd_msgs_.size() == this->GetWorkerNum()) {
    SendActivations();
    fwd_msgs_.clear();
  }
}

template <typename Dtype>
void ConvParamThread<Dtype>::ProcessBackward(shared_ptr<Msg> m)
{
  int conv_id = m->conv_id();
  
  shared_ptr<vector<shared_ptr<Msg> > > bwd_msgs;

  unordered_map<int, shared_ptr<vector<shared_ptr<Msg> > > >::iterator back_iter = 
        bwd_id_to_vec_.find(conv_id);
  
  if (back_iter == bwd_id_to_vec_.end()) {
    bwd_msgs.reset(new vector<shared_ptr<Msg> >());
    bwd_msgs->push_back(m);
    bwd_id_to_vec_[conv_id] = bwd_msgs;
  } else {
    bwd_msgs = back_iter->second;
    bwd_msgs->push_back(m);
  }

  if (bwd_msgs->size() < gateway_ids_.size()) {
    return;
  }

  unordered_map<int, shared_ptr<vector<shared_ptr<Msg> > > >::iterator iter = 
        conv_id_to_vec_.find(conv_id);
  
  CHECK(iter != conv_id_to_vec_.end());
  shared_ptr<vector<shared_ptr<Msg> > > pfwd_msg = iter->second;

  vector<shared_ptr<Net<Dtype> > > net_vec;
  for (int i = 0; i < pfwd_msg->size(); i++) {
    WorkerSolver<Dtype> *pconv = ((WorkerSolver<Dtype> **)pfwd_msg->at(i)->ZmsgData(0))[0];
    net_vec.push_back(pconv->net());
  }
  
  for (int i = 0; i < bwd_msgs->size(); i++) {
    shared_ptr<Msg> bmsg = bwd_msgs->at(i);
    ParamHelper<Dtype>::CopyOutputDiffFromMsg(net_vec, bmsg);
  }

  // swap and send back the buffered messages
  for (int i = 0; i < pfwd_msg->size(); i++) {
    shared_ptr<Msg> m = pfwd_msg->at(i);
    m->set_type(BACKWARD);
    int src = m->src();
    m->set_src(m->dst());
    m->set_dst(src);

    this->SendMsg(m);
  }
  
  conv_id_to_vec_.erase(iter);

  back_iter = bwd_id_to_vec_.find(conv_id);
  CHECK(back_iter != bwd_id_to_vec_.end());
  bwd_id_to_vec_.erase(back_iter);
}

template <typename Dtype>
int ConvParamThread<Dtype>::PutGradient(shared_ptr<Msg> m)
{
  WorkerSolver<Dtype> *psolver = (WorkerSolver<Dtype> *)NodeEnv::Instance()->FindSolver(m->conv_id());
  CHECK(psolver != NULL) << "cannot find solver for conv id: " << m->conv_id();
  
  SGDSolver<Dtype> *root_solver = (SGDSolver<Dtype> *) NodeEnv::Instance()->GetRootSolver();
  shared_ptr<Net<Dtype> > root_net = root_solver->net();
 
  int layer_id = ((int *)m->ZmsgData(0))[0];

  ParamHelper<Dtype>::AddDiffFromNet(root_net, psolver->net(), layer_id);
  layer_updates_[layer_id]++;
  
  if (layer_updates_[layer_id] == this->GetWorkerNum()) {
    SyncLayer(layer_id);
    layer_updates_[layer_id] = 0;
  }

  if (layer_id == 0) {
    NodeEnv::Instance()->DeleteSolver(m->conv_id());
    NodeEnv::Instance()->PushFreeSolver(psolver);
  }

  return 0;
}

template <typename Dtype>
int ConvParamThread<Dtype>::UpdateParam(shared_ptr<Msg> m)
{
  map<int, int>::iterator map_iter = ps_id_map_.find(m->src());
  CHECK(map_iter != ps_id_map_.end());

  SGDSolver<Dtype> *root_solver = (SGDSolver<Dtype> *) NodeEnv::Instance()->GetRootSolver();
  shared_ptr<Net<Dtype> > root_net = root_solver->net();
  
  ParamHelper<Dtype>::CopyParamDataFromMsg(root_net, m);
  
  num_param_update_++;

  if (num_param_update_ < ps_ids_.size()) {
    return -1;
  }

  
  // we have got all the responces from parameter servers
  // and reset the counter
  num_param_update_ = 0;
  
  // clear the gradients of root net
  root_net->ClearParamDiffs();

  // notify the worker threads
  shared_ptr<Msg> notify(new Msg());
  notify->set_type(PUT_PARAM);
  notify->set_src(ROOT_THREAD_ID);
  notify->set_dst(WORKER_BCAST);

  notify->AppendData(&num_param_update_, sizeof(num_param_update_));
  this->SendMsg(notify);

  return 0;
}

template <typename Dtype>
void ConvParamThread<Dtype>::Run()
{
  #ifdef USE_MKL
  mkl_set_dynamic(false);
  #endif
  Caffe::set_root_solver(true);
  
  SGDSolver<Dtype> *root_solver = (SGDSolver<Dtype> *) NodeEnv::Instance()->GetRootSolver();
  shared_ptr<Net<Dtype> > root_net = root_solver->net();
  root_net->ClearParamDiffs();

  // reset the update map
  layer_updates_.resize(root_net->layers().size());
  for (int i = 0; i < layer_updates_.size(); i++) {
    layer_updates_[i] = 0;
  }

  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);

    if (m->type() == PUT_GRADIENT) {
      PutGradient(m);
    } else if (m->type() == PUT_PARAM) {
      UpdateParam(m);
    } else if (m->type() == FORWARD) {
      ProcessForward(m);
    } else if (m->type() == BACKWARD) {
      ProcessBackward(m);
    } else {
      LOG(ERROR) << "PS client: unknown type " << m->type();
    }
  }
  
}

INSTANTIATE_CLASS(ConvThread);
INSTANTIATE_CLASS(ConvParamThread);

} // end namespace caffe


