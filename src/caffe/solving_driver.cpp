#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/solving_driver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
SolvingDriver<Dtype>::SolvingDriver(const SolverParameter& param) {
  Init(param);
}

template <typename Dtype>
SolvingDriver<Dtype>::SolvingDriver(const string& param_file) {
  SolverParameter param;
  ReadProtoFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
SolvingDriver<Dtype>::SolvingDriver(shared_ptr<Solver<Dtype> > solver)
  : solver_(solver) {}

template <typename Dtype>
void SolvingDriver<Dtype>::Init(const SolverParameter& param) {
  solver_.reset(GetSolver<Dtype>(param));
  InitTestNets();
}

template <typename Dtype>
void SolvingDriver<Dtype>::InitTrainNet() {
  // placeholder to match past interface
}

template <typename Dtype>
void SolvingDriver<Dtype>::InitTestNets() {
  const SolverParameter& param_ = solver_->param();
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
      if (has_net_param) {
        for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
          sources[test_net_id] = "net_param";
          net_params[test_net_id].CopyFrom(param_.net_param());
        }
      }
      if (has_net_file) {
        for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
          sources[test_net_id] = "net file: " + param_.net();
          ReadNetParamsFromTextFileOrDie(param_.net(),
                                         &net_params[test_net_id]);
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
      }
    }
  }
}

template <typename Dtype>
void SolvingDriver<Dtype>::Step(int iters) {
  const int start_iter = solver_->iter();
  const int stop_iter = start_iter + iters;
  const SolverParameter& param_ = solver_->param();
  Net<Dtype>* net_ = solver_->net().get();
  for (int iter_ = start_iter; iter_ < stop_iter; ++iter_) {
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {
      TestAll();
    }
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());

    vector<shared_ptr<SolverResult<Dtype> > > output_info;
    solver_->Next(display ? &output_info : NULL);
    if (display) {
      int score_index = 0;
      LOG(INFO) << "Iteration " << solver_->iter()
                << ", loss = " << solver_->smoothed_loss();
      for (int i = 0; i < output_info.size(); ++i) {
        const SolverResult<Dtype>& sr = *(output_info[i]);
        for (int j = 0; j < sr.blob_data.size(); ++j) {
          ostringstream loss_msg_stream;
          if (sr.loss_weight) {
            loss_msg_stream << " (* " << sr.loss_weight
                            << " = " << sr.loss_weight * sr.blob_data[j]
                            << " loss)";
          }
          LOG(INFO) << "    Train net output #"
                    << score_index++ << ": " << sr.blob_name << " = "
                    << sr.blob_data[j] << loss_msg_stream.str();}
      }
    }
    if (param_.snapshot() && (iter_ + 1) % param_.snapshot() == 0) {
      solver_->Snapshot();
    }
  }
}

template <typename Dtype>
void SolvingDriver<Dtype>::Solve(const char* resume_file) {
  Caffe::set_phase(Caffe::TRAIN);
  LOG(INFO) << "Solving " << solver_->net()->name();
  LOG(INFO) << "Learning Rate Policy: " << solver_->param().lr_policy();

  const SolverParameter& param_ = solver_->param();
  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    solver_->Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  Step(param_.max_iter() - solver_->iter());
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || solver_->iter() % param_.snapshot() != 0)) {
    Snapshot();
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && solver_->iter() % param_.display() == 0) {
    Dtype loss;
    solver_->net()->ForwardPrefilled(&loss);
    LOG(INFO) << "Iteration " << solver_->iter() << ", loss = " << loss;
  }
  if (param_.test_interval() && solver_->iter() % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void SolvingDriver<Dtype>::TestAll() {
  for (int test_net_id = 0; test_net_id < test_nets_.size(); ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
void SolvingDriver<Dtype>::Test(const int test_net_id) {
  LOG(INFO) << "Iteration " << solver_->iter()
            << ", Testing net (#" << test_net_id << ")";
  // We need to set phase to test before running.
  Caffe::set_phase(Caffe::TEST);
  const SolverParameter& param_ = param();
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(solver_->net().get());
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
  Caffe::set_phase(Caffe::TRAIN);
}

INSTANTIATE_CLASS(SolvingDriver);

}  // namespace caffe
