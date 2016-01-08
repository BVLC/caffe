#include <string>

#include "caffe/util/io.hpp"
#include "caffe/util/solver_trace.hpp"

namespace caffe {

template <typename Dtype>
SolverTrace<Dtype>::SolverTrace(const SolverParameter& param,
    const Solver<Dtype>* solver)
  : trace_digest_(new TraceDigest()), solver_(solver) {
  param_ = param;
  if (param_.has_solver_trace_param()) {
    trace_param_ = param_.solver_trace_param();
    if (trace_param_.has_save_interval()) {
      CHECK_GE(trace_param_.save_interval(), 0)
        << "solver trace trace save_interval >= 0";
      save_trace_ = trace_param_.save_interval();
    } else {
      save_trace_ = param_.snapshot();
    }
  }
}

template <typename Dtype>
const TraceDigest& SolverTrace<Dtype>::get_digest() const {
  return *trace_digest_;
}

template <typename Dtype>
void SolverTrace<Dtype>::update_trace_train(SolverAction::Enum request) {
  // Continue only if the trace params have been set
  if (!param_.has_solver_trace_param()) {
    return;
  }
  CHECK(Caffe::root_solver());
  int iter = solver_->iter();
  if (iter % save_trace_ == 0 || request == SolverAction::SNAPSHOT) {
    Save();
  }
}

template <typename Dtype>
void SolverTrace<Dtype>::update_trace_test_loss(int test_net_id, Dtype loss) {
  // Continue only if the trace params have been set
  if (!param_.has_solver_trace_param()) {
    return;
  }
  CHECK(Caffe::root_solver());
}

template <typename Dtype>
void SolverTrace<Dtype>::update_trace_test_score(int test_net_id,
                                                 const string& output_name,
                                                 Dtype loss_weight,
                                                 Dtype mean_score) {
  // Continue only if the trace params have been set
  if (!param_.has_solver_trace_param()) {
    return;
  }
  CHECK(Caffe::root_solver());
}

template <typename Dtype>
void SolverTrace<Dtype>::Save() const {
  string filename;
  if (trace_param_.has_trace_filename()) {
    filename = trace_param_.trace_filename() + ".caffetrace";
  } else {
    filename = param_.snapshot_prefix() + ".caffetrace";
  }
  LOG(INFO) << "Snapshotting trace to " << filename;
  WriteProtoToBinaryFile(*trace_digest_, filename.c_str());
  if (trace_param_.human_readable_trace()) {
    filename += "_txt";
    LOG(INFO) << "Snapshotting human readable trace to " << filename;
    WriteProtoToTextFile(*trace_digest_, filename.c_str());
  }
}

INSTANTIATE_CLASS(SolverTrace);

}  // namespace caffe
