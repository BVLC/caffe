#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <vector>
#include "caffe/MultiSolver.hpp"

namespace caffe {

template <typename Dtype>
MultiSolver<Dtype>::MultiSolver(shared_ptr<Solver<Dtype> > root_solver)
  : root_solver_(root_solver) {
  root_solver->set_forward_backward(
    boost::bind(&MultiSolver<Dtype>::ForwardBackward, this));

  worker_solvers.push_back(root_solver_);

  for (int i = 1; i < root_solver->param().iter_size(); ++i) {
    add_another();
  }
}

template <typename Dtype>
int MultiSolver<Dtype>::num_of_solvers() {
  boost::recursive_mutex::scoped_lock lock(mtx);
  return worker_solvers.size();
}

template <typename Dtype>
void MultiSolver<Dtype>::pop_if_able() {
  boost::recursive_mutex::scoped_lock lock(mtx);
  if (worker_solvers.size() > 1) {
    worker_solvers.pop_back();
  }
}

template <typename Dtype>
void MultiSolver<Dtype>::add_another() {
  boost::recursive_mutex::scoped_lock lock(mtx);

  Caffe::set_solver_count(Caffe::solver_count() + 1);
  Caffe::set_root_solver(false);
  CHECK_GT(worker_solvers.size(), 0);

  const vector<shared_ptr<Layer<Dtype> > >& this_layers =
    root_solver_->net()->layers();

  worker_solvers.push_back(
    boost::make_shared<WorkerSolver<Dtype> >(
      root_solver_->param(), root_solver_.get()));

  const vector<shared_ptr<Layer<Dtype> > >& curr_layers =
    worker_solvers.back()->net()->layers();
  CHECK(this_layers.size() == curr_layers.size());
  for (int j = 0; j < curr_layers.size(); ++j) {
    CHECK(this_layers[j]->blobs().size() == curr_layers[j]->blobs().size());

    for (int k = 0; k < this_layers[j]->blobs().size(); ++k) {
      curr_layers[j]->blobs()[k]->ShareData(
        *this_layers[j]->blobs()[k]);
      curr_layers[j]->blobs()[k]->ShareDiff(
        *this_layers[j]->blobs()[k]);
    }
  }
  worker_solvers.back()->net()->ClearParamDiffs();

  Caffe::set_root_solver(true);
}

template <typename Dtype>
Dtype MultiSolver<Dtype>::ForwardBackward() {
  boost::recursive_mutex::scoped_lock lock(mtx);

  Dtype loss = 0;
  Net<Dtype>& net = *root_solver_->net();

  // workers share all params

  for (int i = 0; i < net.layers().size(); ++i) {
    for (int j = 0; j < callbacks_.size(); ++j) {
      callbacks_[j]->on_start(i);
    }
    vector<int> param_ids = net.get_layer_learnable_param_ids(i);
    for (int j = 0; j < worker_solvers.size(); ++j) {
      loss += worker_solvers[j]->net()->ForwardFromTo(i, i);
    }
    for (int j = 0; j < callbacks_.size(); ++j) {
      callbacks_[j]->on_forward_finished(i);
    }
  }

  for (int i = net.layers().size() - 1; i >= 0; --i) {
    for (int j = 0; j < callbacks_.size(); ++j) {
      callbacks_[j]->on_backward_start(i);
    }
    for (int j = 0; j < worker_solvers.size(); ++j) {
      worker_solvers[j]->net()->BackwardFromTo(i, i);
    }
    for (int j = 0; j < callbacks_.size(); ++j) {
      callbacks_[j]->on_gradients_ready(i);
    }
  }

  return loss;
}

template <typename Dtype>
void MultiSolver<Dtype>::Solve() {
  root_solver_->Solve();
}

INSTANTIATE_CLASS(MultiSolver);

}  // namespace caffe

