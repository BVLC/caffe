

#ifndef MULTI_NODE_SOLVER_POOL_HPP_
#define MULTI_NODE_SOLVER_POOL_HPP_

#include <boost/unordered_map.hpp>

#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"

using boost::unordered_map;

namespace caffe {

/*
 * WARNING: the solver pool is NOT thread safe
 */

template <typename Dtype>
class SolverPool {
 public:
  SolverPool() { }

  virtual ~SolverPool() { }

  // Push a free solver to the pool
  void PushFreeSolver(Solver<Dtype> *p);

  // Pop a free solver from solver pool
  // return NULL if no free solvers available
  Solver<Dtype> *PopFreeSolver();

  // Find a solver according to msg id
  Solver<Dtype> *FindSolver(int64_t msg_id);

  // Put a solver to the pool and associate it with a msg id
  void BindSolver(Solver<Dtype> *psolver, int64_t msg_id);

  // release the solver associated with msg_id
  // and put it to free solver pool
  void ReleaseSolver(int64_t msg_id);

  // remove the bind of a solver with message id
  // but don't delete the solver
  void RemoveBind(int64_t msg_id);

 protected:
  unordered_map<int64_t, Solver<Dtype> *> msg_id_to_solver_;

  typedef typename unordered_map<int64_t, Solver<Dtype> *>::iterator
                                                unorder_map_iterator_t;

  // free solver pool
  vector<Solver<Dtype> *> free_solvers_;

DISABLE_COPY_AND_ASSIGN(SolverPool);
};

}  // end namespace caffe

#endif

