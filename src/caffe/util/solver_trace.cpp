#include <string>
#include <vector>
#include "caffe/util/io.hpp"
#include "caffe/util/solver_trace.hpp"

namespace caffe {

template <typename Dtype>
SolverTrace<Dtype>::SolverTrace(const SolverParameter& param,
    const Solver<Dtype>* solver)
  : trace_digest_(new TraceDigest()), solver_(solver), last_iter_(-1) {
  param_ = param;
  if (param_.has_solver_trace_param()) {
    trace_param_ = param_.solver_trace_param();
    if (trace_param_.has_save_interval()) {
      CHECK_GT(trace_param_.save_interval(), 0)
        << "solver trace trace save_interval > 0";
      save_trace_ = trace_param_.save_interval();
    } else {
      CHECK_GT(param_.snapshot(), 0)
        << "param_.snapshot() must be > 0 "
        << "or set a solver trace trace save_interval > 0";
      save_trace_ = param_.snapshot();
    }
    // TODO: update this when the load from historical trace works
    start_iter_ = solver_->iter();

    // Figure out filename where solver trace will be saved
    if (trace_param_.has_trace_filename()) {
      filename_ = trace_param_.trace_filename() + ".caffetrace";
    } else {
      CHECK(param_.has_snapshot_prefix()) <<
          "either snapshot_prefix or trace_filename must be set";
      filename_ = param_.snapshot_prefix() + ".caffetrace";
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
  // If we haven't already saved this iteration
  if (iter != last_iter_) {
    last_iter_ = iter;
    update_weight_trace();
  }
  // this prevents saving the same iteration twice even if we are requested to
  if (iter % save_trace_ == 0 ||
      (request == SolverAction::SNAPSHOT && iter % save_trace_ != 0)) {
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
  if (param_.has_solver_trace_param()) {
    LOG(INFO) << "Snapshotting trace to " << filename_;
    WriteProtoToBinaryFile(*trace_digest_, filename_.c_str());
    if (trace_param_.human_readable_trace()) {
      string hr_filename = filename_ + "_txt";
      LOG(INFO) << "Snapshotting human readable trace to " << hr_filename;
      WriteProtoToTextFile(*trace_digest_, hr_filename.c_str());
    }
  }
}

template <typename Dtype>
void SolverTrace<Dtype>::update_weight_trace() {
  int iter = solver_->iter();
  // If we are at the very start or at a point where we want to create trace
  if ((trace_param_.weight_trace_interval() > 0)
      && ((iter + start_iter_) % trace_param_.weight_trace_interval() == 0)
      && (start_iter_ == 0 || iter > start_iter_)) {
    const vector<shared_ptr<Layer<Dtype> > >& layers= solver_->net()->layers();
    const vector<string>& layer_names = solver_->net()->layer_names();

    // for each layer in the net weight traces are created
    for (int layer_id = 0; layer_id < layers.size(); ++layer_id) {
      const vector<shared_ptr<Blob<Dtype> > >& blobs =
          layers[layer_id]->blobs();
      // for each blob in the layer weight traces are created
      for (int param_id = 0; param_id  < blobs.size(); ++param_id) {
        WeightTracePoint* new_point = trace_digest_->add_weight_trace_point();
        new_point->set_iter(iter);
        new_point->set_layer_name(layer_names[layer_id]);
        new_point->set_param_id(param_id);
        int count = blobs[param_id]->count();
        const Dtype* data = blobs[param_id]->cpu_data();
        // If the blob has a lot of values, add them to trace at even intervals
        if (count > trace_param_.num_weight_traces()) {
          int start = count / (trace_param_.num_weight_traces() * 2 + 2);
          int step  = count / (trace_param_.num_weight_traces() + 1);
          for (int i = 0; i < trace_param_.num_weight_traces(); ++i) {
            new_point->add_value(data[start + i * step]);
          }
        } else {
          // If there are not very many values, add them all to trace
          for (int i = 0; i < count; ++i) {
            new_point->add_value(data[i]);
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(SolverTrace);

}  // namespace caffe
