#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>
#include <stdlib.h>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/srelu_names.hpp"

namespace caffe {

using std::map;
using std::pair;
//using boost::make_shared;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::tuple;
using std::make_tuple;
using std::tie;
using std::get;
using std::to_string;

//const int   SRELU_LAYER_NUM   = 9;
const float srelu_pt_ratio    = 0.9;
const int   srelu_show_gap    = 10000;
const int   srelu_gate_down   = 600000;
const int   srelu_gate_up     = 300000;
const int   srelu_alter       = 10000;

template <typename Dtype>
void Solver<Dtype>::display_srelu() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  for (int n=0; n<SRELU_NAMES.size(); n++) {
    if (n%4 != 0)
      continue;
    LOG(INFO) << SRELU_NAMES[n];
    const int index = this->net_->param_names_index().at(SRELU_NAMES[n]);
    const shared_ptr<Blob<Dtype> >& blob = net_params[index];
    const Dtype* data = blob->cpu_data();
    Dtype sum = 0;
    for (int id=0; id<blob->count(); id++)
      sum += data[id];

    // std::cout << data[id] << " ";
    std::cout << sum/(Dtype)blob->count() << "  ";
  }
}

template <typename Dtype>
void Solver<Dtype>::ResetPreDiff() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  for (int n=0; n<net_params.size(); n++) {
    const shared_ptr<Blob<Dtype> >& blob = net_params[n];
    Dtype* history = ((SGDSolver<Dtype>*)this)->history()[n]->mutable_cpu_data();
    for (int id=0; id<blob->count(); id++) {
      history[id] = 0.;
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::fix_srelu() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  // for ALL
  for (int n=0; n<SRELU_NAMES.size(); n++) {
    const int index = this->net_->param_names_index().at(SRELU_NAMES[n]);
    const shared_ptr<Blob<Dtype> >& blob = net_params[index];
    Dtype* data_diff = blob->mutable_cpu_diff();
    Dtype* history = ((SGDSolver<Dtype>*)this)->history()[index]->mutable_cpu_data();
    for (int id=0; id<blob->count(); id++) {
      data_diff[id] = 0.;
      history[id]   = 0.;
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::adjust_srelu() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  if (this->in_pslope_ == 1) {
    // for pslope
    for (int n=0; n<SRELU_PSLOPE_NAMES.size(); n++) {
      const int index = this->net_->param_names_index().at(SRELU_PSLOPE_NAMES[n]);
      const shared_ptr<Blob<Dtype> >& blob = net_params[index];
      Dtype* data_diff = blob->mutable_cpu_diff();
      Dtype* history = ((SGDSolver<Dtype>*)this)->history()[index]->mutable_cpu_data();
      for (int id=0; id<blob->count(); id++) {
        data_diff[id] = 0.;
        history[id]   = 0.;
      }
    }
  } else {
    // for thresh
    for (int n=0; n<SRELU_THRESH_NAMES.size(); n++) {
      const int index = this->net_->param_names_index().at(SRELU_THRESH_NAMES[n]);
      const shared_ptr<Blob<Dtype> >& blob = net_params[index];
      Dtype* data_diff = blob->mutable_cpu_diff();
      Dtype* history = ((SGDSolver<Dtype>*)this)->history()[index]->mutable_cpu_data();
      for (int id=0; id<blob->count(); id++) {
        data_diff[id] = 0.;
        history[id]   = 0.;
      }
    }
  }
}
template <typename Dtype>
void Solver<Dtype>::Display_param() {
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)
    : net_(), callbacks_(), iteration_timer_(), iterations_last_() {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file)
    : net_(), callbacks_(), iteration_timer_(), iterations_last_() {
  SolverParameter param;
  ReadProtoFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  if (Caffe::root_solver()) {
    LOG(INFO) << "Initializing solver from parameters: " << std::endl
              << param.DebugString();
  }
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  if (Caffe::root_solver()) {
    InitTestNets();
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating training net specified in train_net_param.";
    }
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating training net from train_net file: "
                << param_.train_net();
    }
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating training net specified in net_param.";
    }
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating training net from net file: " << param_.net();
    }
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  net_.reset(new Net<Dtype>(net_param));
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  CHECK(Caffe::root_solver());
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  vector<Blob<Dtype>*> bottom_vec;
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  vector<Dtype> losses;
  Dtype smoothed_loss = 0;

  iteration_timer_.Start();
  Timer timer;
  ostringstream timing;

  while (iter_ < stop_iter) {
    // zero-init the params
    for (int i = 0; i < net_->params().size(); ++i) {
      shared_ptr<Blob<Dtype> > blob = net_->params()[i];
      switch (Caffe::mode()) {
      case Caffe::CPU:
        caffe_set(blob->count(), static_cast<Dtype>(0),
            blob->mutable_cpu_diff());
        break;
      case Caffe::GPU:
#ifndef CPU_ONLY
        caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
            blob->mutable_gpu_diff());
#else
        NO_GPU;
#endif
        break;
      }
    }

    int my_interval;
    my_interval = param_.test_interval();
    if (param_.test_interval() && iter_ % my_interval == 0
        && (iter_ > 0 || param_.test_initialization())
        && Caffe::root_solver()) {
      TestAll();
    }

    timer.Start();
    timing.str("");
    timing << "Timing ";
    if (param().solver_mode() == SolverParameter_SolverMode_GPU) {
      timing << "(device " << param().device_id() << ") ";
    }
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start(&timer, &timing);
    }
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    Dtype loss = 0;
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward(bottom_vec);
    }
    loss /= param_.iter_size();
    // average the loss across iterations for smoothed reporting
    if (losses.size() < average_loss) {
      losses.push_back(loss);
      int size = losses.size();
      smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
    } else {
      int idx = (iter_ - start_iter) % average_loss;
      smoothed_loss += (loss - losses[idx]) / average_loss;
      losses[idx] = loss;
    }
    if (display) {
      if (Caffe::root_solver()) {
        LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss;
      }
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          if (Caffe::root_solver()) {
            LOG(INFO) << "    Train net output #"
                << score_index++ << ": " << output_name << " = "
                << result_vec[k] << loss_msg_stream.str();
          }
        }
      }
    }
    timing << " grads: " << timer.MilliSeconds();
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready(&timer, &timing);
    }
    timer.Start();
    ApplyUpdate();
    timing << " apply: " << timer.MilliSeconds();

#ifdef BENCHMARK_SOLVER
    LOG(INFO)<< timing.str();
#endif

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

    // Save a snapshot if needed.
    if (param_.snapshot()
        && iter_ % param_.snapshot() == 0
        && Caffe::root_solver()) {
      Snapshot();
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    Dtype loss;
    net_->ForwardPrefilled(&loss);
    LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}


template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0; test_net_id < test_nets_.size(); ++test_net_id) {
    Test(test_net_id);
  }
  //Test(1);
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(bottom_vec, &iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}


template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  NetParameter net_param;
  // For intermediate results, we will also dump the gradient values.
  net_->ToProto(&net_param, param_.snapshot_diff());
  string filename(param_.snapshot_prefix());
  string model_filename, snapshot_filename;
  const int kBufferSize = 20;
  char iter_str_buffer[kBufferSize];
  snprintf(iter_str_buffer, kBufferSize, "_iter_%d", iter_);
  filename += iter_str_buffer;
  model_filename = filename + ".caffemodel";
  LOG(INFO) << "Snapshotting to " << model_filename;
  WriteProtoToBinaryFile(net_param, model_filename.c_str());
  SolverState state;
  SnapshotSolverState(&state);
  state.set_iter(iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(current_step_);
  snapshot_filename = filename + ".solverstate";
  LOG(INFO) << "Snapshotting solver state to " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  SolverState state;
  NetParameter net_param;
  ReadProtoFromBinaryFile(state_file, &state);
  if (state.has_learned_net()) {
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    net_->CopyTrainedLayersFrom(net_param);
  }
  iter_ = state.iter();
  current_step_ = state.current_step();
  RestoreSolverState(state);
}


// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    if (this->net_->param_owners()[i] < 0) {
      sumsq_diff += net_params[i]->sumsq_diff();
    }
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      if (this->net_->param_owners()[i] < 0) {
        net_params[i]->scale_diff(scale_factor);
      }
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::topk_heap(vector<Dtype>& score, Dtype& thresh, const Dtype prob) {
  int data_num = score.size();
  int heap_size = prob * (Dtype)data_num;
  if (!heap_size) {
    thresh = 0.;
    return;
  }
  LOG(INFO) << "THE NUMS ARE: " << data_num << "        " << heap_size;
  vector<Dtype> tmp(heap_size, 0.);
  for (int i=0; i<heap_size; i++) {
    tmp.at(i) = fabs(score[i]);
  }
  make_heap(tmp.begin(), tmp.end());
  for (int i=heap_size; i<data_num; i++) {
    if (fabs(score[i]) < *tmp.begin()) {
      pop_heap( tmp.begin(), tmp.end() );
      tmp.pop_back();
      tmp.push_back( fabs(score[i]) );
      push_heap( tmp.begin(), tmp.end() );
    }
  }
  make_heap( tmp.begin(), tmp.end() );
  thresh = *tmp.begin();
}

template <typename Dtype>
void Solver<Dtype>::srelu_pt_adapt() {
  CHECK(Caffe::root_solver());
  bool shared_ = 1;
  CHECK_NOTNULL(test_nets_[0].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[0];
  const int ptNum = SRELU_NAMES.size()/4;
  //CHECK_EQ(ptNum, SRELU_LAYER_NUM); // For NON-googlenet
  vector<vector<vector<Dtype> > > pt_channel_in( ptNum );
  vector<vector<Dtype> > pt_channel_out( ptNum );
  vector<shared_ptr<Layer<Dtype> > > srelu_layers( ptNum );
  vector<shared_ptr<Blob<Dtype> > > srelu_top_blob( ptNum );
  vector<shared_ptr<Blob<Dtype> > > srelu_param_blob( ptNum );
  for (int i = 0; i < ptNum; i++) {
    string layer_name_;
    layer_name_.assign(SRELU_NAMES[ i*4 ], 0, SRELU_NAMES[ i*4 ].length() - 7);
    if (test_net->param_names_index().find( SRELU_NAMES[ i*4 ] ) ==
        test_net->param_names_index().end())
      LOG(FATAL) << "No SReLU param named: " << SRELU_NAMES[ i*4 ];
    if (test_net->layer_names_index().find( layer_name_ ) ==
        test_net->layer_names_index().end() )
      LOG(FATAL) << "No SReLU layer named: " << layer_name_;
    shared_ptr<Blob<Dtype> > srelu_blob_ = test_net->params()[
        test_net->param_names_index().find(SRELU_NAMES[i*4])->second ];
    shared_ptr<Layer<Dtype> > srelu_layer_ =
        test_net->layer_by_name( layer_name_ );
    srelu_param_blob[i] = srelu_blob_;
    int channel_ = srelu_blob_->count();
    pt_channel_in[i].resize( channel_ );
    pt_channel_out[i].resize( channel_ );
    srelu_layers[i] = srelu_layer_;
    string srelu_top_blob_name_ = srelu_layer_->layer_param().top(0);
    if (test_net->blob_names_index().find( srelu_top_blob_name_ ) ==
        test_net->blob_names_index().end() )
      LOG(FATAL) << "No SReLU blob named: " << srelu_top_blob_name_;
    srelu_top_blob[i] = test_net->blob_by_name( srelu_top_blob_name_ );
  }
  // collect all samples' channel-wise pt value
  for (int i = 0; i < 1000; ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(bottom_vec, &iter_loss);
    //if (!(i % 5 == 0))
    //  continue;
    for (int s = 0; s < ptNum; s++) {
      int num = srelu_top_blob[s]->num();
      int channel = srelu_top_blob[s]->channels();
      int size = srelu_top_blob[s]->height() * srelu_top_blob[s]->width();
      const Dtype* data = srelu_top_blob[s]->cpu_data();
      for (int n = 0; n < num; n++) {
        for (int c = 0; c < channel; c++) {
          Dtype sum = 0;
          int amount = 0;
          int offset = srelu_top_blob[s]->offset(n,c,0,0);
          for (int count=0; count < size; count++) {
            if (data[offset + count] > 0) {
              sum += data[offset + count];
              amount ++;
            }
          }
          if (amount == 0) {
            ;
            //LOG(INFO) << "Amount is zero" << "ptNum and channel is: " << s << " " << c;
            //pt_channel_in[s][c].push_back(0);
          }
          else
            pt_channel_in[s][c].push_back( sum / (Dtype)amount );
        }
      }
    }
  }
  // calculate the thresh of all samples
  for (int s = 0; s < ptNum; s++) {
    int channel = srelu_top_blob[s]->channels();
    Dtype thresh_;
    for (int c = 0; c < channel; c++) {
      this->topk_heap( pt_channel_in[s][c], thresh_, srelu_pt_ratio);
      if (thresh_ == 0) {
        LOG(INFO) << "Thresh is zero" << "ptNum and channel is: " << s << " " << c;
        pt_channel_out[s][c] = 0;
      } else {
        pt_channel_out[s][c] = thresh_;
      }
    }
  }
  for (int s = 0; s < ptNum; s++) {
    int channel = srelu_top_blob[s]->channels();
    for (int c = 0; c < channel; c++) {
      Dtype left, right;
      // find the neighboring values
      if (pt_channel_out[s][c] == 0) {
        for (int p=0; p < channel; p++)
          if (pt_channel_out[s][(c-p+channel)%channel] != 0 ) {
            left = pt_channel_out[s][(c-p+channel)%channel];
            break;
          }
        for (int p=0; p < channel; p++)
          if (pt_channel_out[s][(c+p)%channel] != 0) {
            right = pt_channel_out[s][(c+p)%channel];
            break;
          }
        if (0.5*left + 0.5*right == 0)
          pt_channel_out[s][c] = 40.;
        else
          pt_channel_out[s][c] = 0.5*left + 0.5*right;
      }
      LOG(INFO) << "The ptNum and channel is: "   << s << " " << c
                << " and The size of vector is: " << pt_channel_in[s][c].size()
                << " and the thresh is: "         << pt_channel_out[s][c];
    }
  }
  // assign initial values to srelu_pt
  for (int s = 0; s < ptNum; s++) {
    Dtype* srelu_pt_init = srelu_param_blob[s]->mutable_cpu_data();
    int count = srelu_param_blob[s]->count();
    LOG(INFO) << "count is: " << count;
    Dtype average = 0;
    for (int c = 0; c < count; c++)
      average += pt_channel_out[s][c];
    average /= (Dtype)count;
    for (int c = 0; c < count; c++) {
      srelu_pt_init[c] = average;
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  CHECK(Caffe::root_solver());
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    float lapse = iteration_timer_.Seconds();
    float per_s = (this->iter_ - iterations_last_) / (lapse ? lapse : 1);
    LOG(INFO) << "Iteration " << this->iter_ << " (" << per_s << "/s), "
              << "lr = " << rate;
    iteration_timer_.Start();
    iterations_last_ = this->iter_;
  }
  ClipGradients();
  for (int param_id = 0; param_id < this->net_->params().size(); ++param_id) {
    Normalize(param_id);
    Regularize(param_id);
    ComputeUpdateValue(param_id, rate);
  }
  if (this->iter_ <= srelu_gate_down) {
    this->fix_srelu();
    if (this->iter_ == srelu_gate_down)
      this->srelu_pt_adapt();
  }
  if (this->iter_ % srelu_show_gap == 0)
    this->display_srelu();
  this->net_->Update();
}

template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  if (this->param_.iter_size() == 1) { return; }
  // Scale gradient to counterbalance accumulation.
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else if (regularization_type == "L1") {
        caffe_cpu_sign(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else if (regularization_type == "L1") {
        caffe_gpu_sign(net_params[param_id]->count(),
            net_params[param_id]->gpu_data(),
            temp_[param_id]->mutable_gpu_data());
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              history_[param_id]->mutable_cpu_data());
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->gpu_diff(), momentum,
              history_[param_id]->mutable_gpu_data());
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->gpu_data(),
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(SolverState* state) {
  state->clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state->add_history();
    history_[i]->ToProto(history_blob);
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK(Caffe::root_solver());
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template <typename Dtype>
void NesterovSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  CHECK(Caffe::root_solver());
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // save history momentum for stepping back
    caffe_copy(net_params[param_id]->count(),
        this->history_[param_id]->cpu_data(),
        this->update_[param_id]->mutable_cpu_data());

    // update history
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              this->history_[param_id]->mutable_cpu_data());

    // compute update: step back then over step
    caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
        this->history_[param_id]->cpu_data(), -momentum,
        this->update_[param_id]->mutable_cpu_data());

    // copy
    caffe_copy(net_params[param_id]->count(),
        this->update_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    // save history momentum for stepping back
    caffe_copy(net_params[param_id]->count(),
        this->history_[param_id]->gpu_data(),
        this->update_[param_id]->mutable_gpu_data());

    // update history
    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->gpu_diff(), momentum,
              this->history_[param_id]->mutable_gpu_data());

    // compute update: step back then over step
    caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
        this->history_[param_id]->gpu_data(), -momentum,
        this->update_[param_id]->mutable_gpu_data());

    // copy
    caffe_copy(net_params[param_id]->count(),
        this->update_[param_id]->gpu_data(),
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void AdaGradSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  CHECK(Caffe::root_solver());
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype delta = this->param_.delta();
  Dtype local_rate = rate * net_params_lr[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // compute square of gradient in update
    caffe_powx(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_cpu_data());

    // update history
    caffe_add(net_params[param_id]->count(),
        this->update_[param_id]->cpu_data(),
        this->history_[param_id]->cpu_data(),
        this->history_[param_id]->mutable_cpu_data());

    // prepare update
    caffe_powx(net_params[param_id]->count(),
              this->history_[param_id]->cpu_data(), Dtype(0.5),
              this->update_[param_id]->mutable_cpu_data());

    caffe_add_scalar(net_params[param_id]->count(),
              delta, this->update_[param_id]->mutable_cpu_data());

    caffe_div(net_params[param_id]->count(),
              net_params[param_id]->cpu_diff(),
              this->update_[param_id]->cpu_data(),
              this->update_[param_id]->mutable_cpu_data());

    // scale and copy
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
        this->update_[param_id]->cpu_data(), Dtype(0),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    // compute square of gradient in update
    caffe_gpu_powx(net_params[param_id]->count(),
        net_params[param_id]->gpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_gpu_data());

    // update history
    caffe_gpu_add(net_params[param_id]->count(),
        this->update_[param_id]->gpu_data(),
        this->history_[param_id]->gpu_data(),
        this->history_[param_id]->mutable_gpu_data());

    // prepare update
    caffe_gpu_powx(net_params[param_id]->count(),
              this->history_[param_id]->gpu_data(), Dtype(0.5),
              this->update_[param_id]->mutable_gpu_data());

    caffe_gpu_add_scalar(net_params[param_id]->count(),
              delta, this->update_[param_id]->mutable_gpu_data());

    caffe_gpu_div(net_params[param_id]->count(),
              net_params[param_id]->gpu_diff(),
              this->update_[param_id]->gpu_data(),
              this->update_[param_id]->mutable_gpu_data());

    // scale and copy
    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
        this->update_[param_id]->gpu_data(), Dtype(0),
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(Solver);
INSTANTIATE_CLASS(SGDSolver);
INSTANTIATE_CLASS(NesterovSolver);
INSTANTIATE_CLASS(AdaGradSolver);

}  // namespace caffe
