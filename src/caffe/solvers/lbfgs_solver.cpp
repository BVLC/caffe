#include <string>
#include <vector>

#include "caffe/lbfgs_solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

// TODO GPU implementation

namespace caffe {

template <typename Dtype>
void LBFGSSolver<Dtype>::PreSolve() {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  n_ = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    if (this->net_->params_lr()[i] != 0) {
      n_ += net_params[i]->count();
    }
  }
  if (n_ == 0) {
    // nothing to do, all learnable parameters have lr_mult = 0
    return;
  }
  const vector<int> shape(1, n_);
  s_history_.clear();
  y_history_.clear();
  rho_history_.clear();
  start_ = 0;
  end_ = -1;
  gradients_prev_.reset(new Blob<Dtype>(shape));
  gradients_.reset(new Blob<Dtype>(shape));
  direction_.reset(new Blob<Dtype>(shape));
  for (int i = 0; i < this->param_.lbfgs_corrections(); ++i) {
    s_history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    y_history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    rho_history_.push_back(0);
  }
}

template <typename Dtype>
void LBFGSSolver<Dtype>::ApplyUpdate() {
  if (n_ == 0) {
    for (int i = 0; i < this->net_->learnable_params().size(); ++i) {
      const int n = this->net_->learnable_params()[i]->count();
      caffe_scal(n, (Dtype)0,
          this->net_->learnable_params()[i]->mutable_cpu_diff());
    }
    return;
  }
  CHECK(Caffe::root_solver());
  CollectGradients();
  UpdateHistory();
  ComputeInitialHessianApprox();
  ComputeDirection();
  ComputeStep();
  UpdateNet();
}

template <typename Dtype>
void LBFGSSolver<Dtype>::CollectGradients() {
  if (this->iter_ != 0) {
    caffe_copy(n_, gradients_->cpu_data(), gradients_prev_->mutable_cpu_data());
  }
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype* data = gradients_->mutable_cpu_data();
  for (int i = 0, j = 0; i < net_params.size(); ++i) {
    if (this->net_->params_lr()[i] != 0) {
      caffe_copy(net_params[i]->count(), net_params[i]->cpu_diff(), &data[j]);
      j += net_params[i]->count();
    }
  }
}

template <typename Dtype>
void LBFGSSolver<Dtype>::UpdateHistory() {
  if (this->iter_ == 0) { return; }
  caffe_scal(n_, -(Dtype)1.0, direction_->mutable_cpu_data());  // s
  caffe_cpu_axpby(n_, (Dtype)1.0, gradients_->cpu_data(),
      -(Dtype)1.0, gradients_prev_->mutable_cpu_data());  // y
  Dtype ys = caffe_cpu_dot(n_, direction_->cpu_data(),
      gradients_prev_->cpu_data());
  if (ys < 1e-10) {
    LOG(INFO) << "Skipped L-BFGS update";
    return;
  }
  end_ += 1;
  if (end_ < this->param_.lbfgs_corrections()) {
    if (start_ != 0) {
      start_ += 1;
      if (start_ == this->param_.lbfgs_corrections()) {
        start_ = 0;
      }
    }
  } else {
    start_ = 1;
    end_ = 0;
  }
  caffe_copy(n_, direction_->cpu_data(), s_history_[end_]->mutable_cpu_data());
  caffe_copy(n_, gradients_prev_->cpu_data(),
      y_history_[end_]->mutable_cpu_data());
  rho_history_[end_] = 1 / ys;
}

template <typename Dtype>
void LBFGSSolver<Dtype>::ComputeInitialHessianApprox() {
  if (this->iter_ == 0) { return; }
  h0_ = 1 / rho_history_[end_] / caffe_cpu_dot(n_,
      y_history_[end_]->cpu_data(), y_history_[end_]->cpu_data());
}

const vector<int> lbfgs_history_indices(int start, int end, int max) {
  vector<int> indices(start == 0 ? end+1 : max);
  if (start == 0) {
    for (int i = start; i <= end; ++i) {
      indices[i] = i;
    }
  } else {
    int j = 0;
    for (int i = start; i < indices.size(); ++i) {
      indices[j++] = i;
    }
    for (int i = 0; i <= end; ++i) {
      indices[j++] = i;
    }
  }
  return indices;
}

template <typename Dtype>
void LBFGSSolver<Dtype>::ComputeDirection() {
  caffe_copy(n_, gradients_->cpu_data(), direction_->mutable_cpu_data());
  if (this->iter_ == 0) { return; }
  const vector<int> indices = lbfgs_history_indices(start_, end_,
      this->param_.lbfgs_corrections());
  vector <Dtype> alpha(indices.size());
  Dtype beta;
  for (int i = indices.size(); i-- > 0;) {
    int idx = indices[i];
    alpha[idx] = rho_history_[idx] * caffe_cpu_dot(n_,
        s_history_[idx]->cpu_data(), direction_->cpu_data());
    caffe_axpy(n_, -alpha[idx], y_history_[idx]->cpu_data(),
        direction_->mutable_cpu_data());
  }
  caffe_scal(n_, h0_, direction_->mutable_cpu_data());
  for (int i = 0; i < indices.size(); ++i) {
    int idx = indices[i];
    beta = rho_history_[idx] * caffe_cpu_dot(n_,
        y_history_[idx]->cpu_data(), direction_->cpu_data());
    caffe_axpy(n_, alpha[idx] - beta, s_history_[idx]->cpu_data(),
        direction_->mutable_cpu_data());
  }
}

template <typename Dtype>
void LBFGSSolver<Dtype>::ComputeStep() {
  // TODO Wolfe line search
  step_ = 1.0;
}

template <typename Dtype>
void LBFGSSolver<Dtype>::UpdateNet() {
  caffe_scal(n_, step_, direction_->mutable_cpu_data());
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const Dtype* dir = direction_->cpu_data();
  for (int i = 0, j = 0, n; i < net_params.size(); ++i) {
    n = net_params[i]->count();
    if (this->net_->params_lr()[i] != 0) {
      caffe_cpu_scale(n, (Dtype)this->net_->params_lr()[i], &dir[j],
          net_params[i]->mutable_cpu_diff());
      j += n;
    } else {
      caffe_scal(n, (Dtype)0, net_params[i]->mutable_cpu_diff());
    }
  }
  this->net_->Update();
}

template <typename Dtype>
void LBFGSSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
  switch (this->param_.snapshot_format()) {
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      SnapshotSolverStateToBinaryProto(model_filename);
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      SnapshotSolverStateToHDF5(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}

template <typename Dtype>
void LBFGSSolver<Dtype>::SnapshotSolverStateToBinaryProto(
    const string& model_filename) {
  SolverState state;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.clear_s_history();
  state.clear_y_history();
  state.clear_rho_history();
  const vector<int> indices = lbfgs_history_indices(start_, end_,
      this->param_.lbfgs_corrections());
  for (int i = 0; i < indices.size(); ++i) {
    int idx = indices[i];
    BlobProto* s_history_blob;
    s_history_blob = state.add_s_history();
    s_history_[idx]->ToProto(s_history_blob);
    BlobProto* y_history_blob;
    y_history_blob = state.add_y_history();
    y_history_[idx]->ToProto(y_history_blob);
    state.add_rho_history(rho_history_[idx]);
  }
  BlobProto gradients_blob;
  gradients_->ToProto(&gradients_blob);
  state.mutable_gradients()->CopyFrom(gradients_blob);
  BlobProto dir_blob;
  direction_->ToProto(&dir_blob);
  state.mutable_direction()->CopyFrom(dir_blob);
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
  LOG(INFO)
    << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

template <typename Dtype>
void LBFGSSolver<Dtype>::SnapshotSolverStateToHDF5(
    const string& model_filename) {
  // TODO
}

template <typename Dtype>
void LBFGSSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
  // TODO
}

template <typename Dtype>
void LBFGSSolver<Dtype>::RestoreSolverStateFromBinaryProto(
    const string& state_file) {
  SolverState state;
  ReadProtoFromBinaryFile(state_file, &state);
  this->iter_ = state.iter();
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  for (int i = 0; i < state.s_history_size(); ++i) {
    s_history_[i]->FromProto(state.s_history(i));
    y_history_[i]->FromProto(state.y_history(i));
    rho_history_[i] = state.rho_history(i);
  }
  end_ = state.s_history_size()-1;
  direction_->FromProto(state.direction());
  gradients_->FromProto(state.gradients());
}

INSTANTIATE_CLASS(LBFGSSolver);
REGISTER_SOLVER_CLASS(LBFGS);

}  // namespace caffe
