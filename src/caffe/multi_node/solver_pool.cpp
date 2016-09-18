
#include <string>
#include <vector>

#include "caffe/multi_node/solver_pool.hpp"


namespace caffe {

template <typename Dtype>
void SolverPool<Dtype>::PushFreeSolver(Solver<Dtype> *p) {
  free_solvers_.push_back(p);
}

template <typename Dtype>
Solver<Dtype> *SolverPool<Dtype>::PopFreeSolver() {
  if (free_solvers_.empty()) {
    return NULL;
  }

  Solver<Dtype> *p = free_solvers_.back();
  free_solvers_.pop_back();

  return p;
}

template <typename Dtype>
Solver<Dtype> *SolverPool<Dtype>::FindSolver(int64_t msg_id) {
  unorder_map_iterator_t iter = msg_id_to_solver_.find(msg_id);
  if (iter == msg_id_to_solver_.end()) {
    return NULL;
  } else {
    return iter->second;
  }
}

template <typename Dtype>
void SolverPool<Dtype>::BindSolver(Solver<Dtype> *psolver, int64_t msg_id) {
  unorder_map_iterator_t iter = msg_id_to_solver_.find(msg_id);
  CHECK(iter == msg_id_to_solver_.end());
  msg_id_to_solver_[msg_id] = psolver;
}

template <typename Dtype>
void SolverPool<Dtype>::RemoveBind(int64_t msg_id) {
  unorder_map_iterator_t iter = msg_id_to_solver_.find(msg_id);

  CHECK(iter != msg_id_to_solver_.end());

  msg_id_to_solver_.erase(iter);
}

template <typename Dtype>
void SolverPool<Dtype>::ReleaseSolver(int64_t msg_id) {
  unorder_map_iterator_t iter = msg_id_to_solver_.find(msg_id);

  CHECK(iter != msg_id_to_solver_.end());

  free_solvers_.push_back(iter->second);
  msg_id_to_solver_.erase(iter);
}

INSTANTIATE_CLASS(SolverPool);
}  // end namespace caffe


