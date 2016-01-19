#include <map>
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

    if (trace_param_.has_trace_interval()) {
      CHECK_GE(trace_param_.trace_interval(), 0)
          << "trace_interval must be non negative";
    }

    if (trace_param_.has_num_traces()) {
      CHECK_GE(trace_param_.num_traces(), 0)
          << "num_traces must be non negative";
    }

    init_weight_trace();
    init_activation_trace();
    init_diff_trace();
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
    if (iter > 0) {
      update_diff_trace();
      update_activation_trace();
    }
  }
  // this prevents saving the same iteration twice even if we are requested to
  if (iter % save_trace_ == 0 ||
      (request == SolverAction::SNAPSHOT && iter % save_trace_ != 0)) {
    Save();
  }
}

template <typename Dtype>
void SolverTrace<Dtype>::update_trace_train_loss(Dtype loss,
                                                 Dtype smoothed_loss) {
  CHECK(Caffe::root_solver());
  // Continue only if the trace params have been set
  if (param_.has_solver_trace_param() && trace_param_.create_train_trace()) {
    TrainTracePoint* new_point = trace_digest_->add_train_trace_point();
    new_point->set_iter(solver_->iter());
    new_point->set_train_loss(loss);
    new_point->set_train_smoothed_loss(smoothed_loss);
  }
}

template <typename Dtype>
void SolverTrace<Dtype>::update_trace_test_loss(int test_net_id, Dtype loss) {
  CHECK(Caffe::root_solver());
  if (param_.has_solver_trace_param() && trace_param_.create_test_trace()) {
    TestLossTracePoint* new_point = trace_digest_->add_test_loss_trace_point();
    new_point->set_test_net_id(test_net_id);
    new_point->set_iter(solver_->iter());
    new_point->set_test_loss(loss);
  }
}

template <typename Dtype>
void SolverTrace<Dtype>::update_trace_test_score(int test_net_id,
                                                 const string& output_name,
                                                 Dtype loss_weight,
                                                 Dtype mean_score) {
  CHECK(Caffe::root_solver());
  if (param_.has_solver_trace_param() && trace_param_.create_test_trace()) {
    TestScoreTracePoint* new_point =
        trace_digest_->add_test_score_trace_point();
    new_point->set_test_net_id(test_net_id);
    new_point->set_iter(solver_->iter());
    new_point->set_score_name(output_name);
    new_point->set_mean_score(mean_score);
    new_point->set_loss_weight(loss_weight);
  }
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
void SolverTrace<Dtype>::Restore(const string& trace_filename) {
  // Read the trace digest in from file
  TraceDigest local_digest;
  ReadProtoFromBinaryFileOrDie(trace_filename, &local_digest);
  const int iter = solver_->iter();
  last_iter_ = iter;

  trace_digest_->clear_weight_trace_point();
  for (int i = 0; i < local_digest.weight_trace_point_size(); ++i) {
    if (local_digest.weight_trace_point(i).iter() <= iter) {
      WeightTracePoint* point = trace_digest_->add_weight_trace_point();
      *point = local_digest.weight_trace_point(i);
    }
  }

  trace_digest_->clear_diff_trace_point();
  for (int i = 0; i < local_digest.diff_trace_point_size(); ++i) {
    if (local_digest.diff_trace_point(i).iter() <= iter) {
      WeightTracePoint* point = trace_digest_->add_diff_trace_point();
      *point = local_digest.diff_trace_point(i);
    }
  }

  trace_digest_->clear_activation_trace();
  activation_name_to_index_.clear();
  for (int i = 0; i < local_digest.activation_trace_size(); ++i) {
    ActivationTrace* new_trace = trace_digest_->add_activation_trace();
    const ActivationTrace& old_trace = local_digest.activation_trace(i);
    new_trace->set_blob_name(old_trace.blob_name());
    activation_name_to_index_[old_trace.blob_name()] = i;
    for (int j = 0; j < old_trace.activation_trace_point_size(); ++j) {
      if (old_trace.activation_trace_point(j).iter() <= iter) {
        ActivationTracePoint* atp = new_trace->add_activation_trace_point();
        *atp = old_trace.activation_trace_point(j);
      }
    }
  }

  trace_digest_->clear_train_trace_point();
  for (int i = 0; i < local_digest.train_trace_point_size(); ++i) {
    if (local_digest.train_trace_point(i).iter() <= iter) {
      TrainTracePoint* point = trace_digest_->add_train_trace_point();
      *point = local_digest.train_trace_point(i);
    }
  }

  trace_digest_->clear_test_loss_trace_point();
  for (int i = 0; i < local_digest.test_loss_trace_point_size(); ++i) {
    if (local_digest.test_loss_trace_point(i).iter() <= iter) {
      TestLossTracePoint* point = trace_digest_->add_test_loss_trace_point();
      *point = local_digest.test_loss_trace_point(i);
    }
  }

  trace_digest_->clear_test_score_trace_point();
  for (int i = 0; i < local_digest.test_score_trace_point_size(); ++i) {
    if (local_digest.test_score_trace_point(i).iter() <= iter) {
      TestScoreTracePoint* point = trace_digest_->add_test_score_trace_point();
      *point = local_digest.test_score_trace_point(i);
    }
  }
}

template <typename Dtype>
void SolverTrace<Dtype>::init_weight_trace() {
  // set weight_trace_interval_ to zero as our default value
  weight_trace_interval_ = 0;
  // If weight trace interval is defined, use that
  if (trace_param_.has_weight_trace_interval()) {
    CHECK_GE(trace_param_.weight_trace_interval(), 0)
        << "weight_trace_interval must be greater than or equal to zero";
    weight_trace_interval_ = trace_param_.weight_trace_interval();
  } else {
    if (trace_param_.has_trace_interval()) {
      weight_trace_interval_ = trace_param_.trace_interval();
    }
  }

  // set num_weight_traces_ to zero as our default value
  num_weight_traces_ = 0;
  // If num_weight_traces is defined, use that
  if (trace_param_.has_num_weight_traces()) {
    CHECK_GE(trace_param_.num_weight_traces(), 0)
        << "num_weight_traces must be greater than or equal to zero";
    num_weight_traces_ = trace_param_.num_weight_traces();
  } else {
    if (trace_param_.has_num_traces()) {
      num_weight_traces_ = trace_param_.num_traces();
    }
  }
}

template <typename Dtype>
void SolverTrace<Dtype>::init_diff_trace() {
  // set diff_trace_interval_ to zero as our default value
  diff_trace_interval_ = 0;
  // If weight trace interval is defined, use that
  if (trace_param_.has_weight_trace_interval()) {
    CHECK_GE(trace_param_.diff_trace_interval(), 0)
        << "diff_trace_interval must be greater than or equal to zero";
    diff_trace_interval_ = trace_param_.diff_trace_interval();
  } else {
    if (trace_param_.has_trace_interval()) {
      diff_trace_interval_ = trace_param_.trace_interval();
    }
  }

  // set num_diff_traces_ to zero as our default value
  num_diff_traces_ = 0;
  // If num_diff_traces is defined, use that
  if (trace_param_.num_diff_traces()) {
    CHECK_GE(trace_param_.num_diff_traces(), 0)
        << "num_diff_traces must be greater than or equal to zero";
    num_diff_traces_ = trace_param_.num_diff_traces();
  } else {
    if (trace_param_.has_num_traces()) {
      num_diff_traces_ = trace_param_.num_traces();
    }
  }
}

template <typename Dtype>
void SolverTrace<Dtype>::update_weight_trace() {
  blob_trace(weight_trace_interval_, num_weight_traces_, true);
}

template <typename Dtype>
void SolverTrace<Dtype>::update_diff_trace() {
  blob_trace(diff_trace_interval_, num_diff_traces_, false);
}

template <typename Dtype>
void SolverTrace<Dtype>::blob_trace(int trace_interval,
                                    int num_traces, bool use_data) {
  int iter = solver_->iter();
  // If we are at the very start or at a point where we want to create trace
  if ((trace_interval > 0)
      && ((iter + start_iter_) % trace_interval == 0)
      && (start_iter_ == 0 || iter > start_iter_)) {
    const vector<shared_ptr<Layer<Dtype> > >& layers= solver_->net()->layers();
    const vector<string>& layer_names = solver_->net()->layer_names();

    // for each layer in the net weight traces are created
    for (int layer_id = 0; layer_id < layers.size(); ++layer_id) {
      const vector<shared_ptr<Blob<Dtype> > >& blobs =
          layers[layer_id]->blobs();
      // for each blob in the layer weight traces are created
      for (int param_id = 0; param_id  < blobs.size(); ++param_id) {
        WeightTracePoint* new_point = NULL;
        if (use_data) {
          new_point = trace_digest_->add_weight_trace_point();
        } else {
          new_point = trace_digest_->add_diff_trace_point();
        }
        new_point->set_iter(iter);
        new_point->set_layer_name(layer_names[layer_id]);
        new_point->set_param_id(param_id);
        int count = blobs[param_id]->count();
        const Dtype* data = NULL;
        if (use_data) {
          data = blobs[param_id]->cpu_data();
        } else {
          data = blobs[param_id]->cpu_diff();
        }
        // If the blob has a lot of values, add them to trace at even intervals
        if (count > num_traces) {
          int start = count / (num_traces * 2 + 2);
          int step  = count / (num_traces + 1);
          for (int i = 0; i < num_traces; ++i) {
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

template <typename Dtype>
void SolverTrace<Dtype>::init_activation_trace() {
  // set activation_trace_interval_ to zero as our default value
  activation_trace_interval_ = 0;
  // If activation trace interval is defined, use that
  if (trace_param_.has_activation_trace_interval()) {
    CHECK_GE(trace_param_.activation_trace_interval(), 0)
        << "weight_trace_interval must be greater than or equal to zero";
    activation_trace_interval_ = trace_param_.activation_trace_interval();
  } else {
    if (trace_param_.has_trace_interval()) {
      activation_trace_interval_ = trace_param_.trace_interval();
    }
  }

  // set num_activation_traces_ to zero as our default value
  num_activation_traces_ = 0;
  // If num_weight_traces is defined, use that
  if (trace_param_.has_num_activation_traces()) {
    CHECK_GE(trace_param_.num_activation_traces(), 0)
        << "num_weight_traces must be greater than or equal to zero";
    num_activation_traces_ = trace_param_.num_activation_traces();
  } else {
    if (trace_param_.has_num_traces()) {
      num_activation_traces_ = trace_param_.num_traces();
    }
  }
}

template <typename Dtype>
void SolverTrace<Dtype>::update_activation_trace() {
  int iter = solver_->iter();
  // If we are at the very start or at a point where we want to create trace
  if ((activation_trace_interval_ > 0)
      && ((iter + start_iter_) % activation_trace_interval_ == 0)
      && (start_iter_ == 0 || iter > start_iter_)) {
    const vector<shared_ptr<Blob<Dtype> > >& blobs = solver_->net()->blobs();
    const vector<string>& blob_names = solver_->net()->blob_names();
    int blob_idx;

    // for each blob in the net activation traces are created
    for (int blob_id = 0; blob_id < blobs.size(); ++blob_id) {
      const shared_ptr<Blob<Dtype> > blob = blobs[blob_id];
      string name = blob_names[blob_id];

      // check to see if a trace for this blob has already been started
      std::map<string, int>::iterator it =activation_name_to_index_.find(name);
      if (it == activation_name_to_index_.end()) {
        ActivationTrace* new_trace = trace_digest_->add_activation_trace();
        new_trace->set_blob_name(name);
        blob_idx = trace_digest_->activation_trace_size()-1;
        activation_name_to_index_[name] = blob_idx;
      } else {
        blob_idx = it->second;
      }
      ActivationTrace* mat = trace_digest_->mutable_activation_trace(blob_idx);
      ActivationTracePoint* new_point = mat->add_activation_trace_point();

      new_point->set_iter(iter);
      int count = blob->count();
      const Dtype* data = blob->cpu_data();
      // If the blob has a lot of values, add them to trace at even intervals
      if (count > num_activation_traces_) {
        int start = count / (num_activation_traces_ * 2 + 2);
        int step  = count / (num_activation_traces_ + 1);
        for (int i = 0; i < num_activation_traces_; ++i) {
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

INSTANTIATE_CLASS(SolverTrace);

}  // namespace caffe
