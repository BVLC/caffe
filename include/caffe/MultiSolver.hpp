#ifndef CAFFE_MULTISOLVER_HPP_
#define CAFFE_MULTISOLVER_HPP_

#include <boost/function.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver.hpp"
#include "caffe/solver_factory.hpp"

namespace caffe {

template <typename Dtype>
class MultiSolver {
 public:
  explicit MultiSolver(shared_ptr<Solver<Dtype> > root_solver);

  // Invoked at specific points during an iteration
  class Callback : public Solver<Dtype>::Callback {
   protected:
    virtual void on_start(int layer_id) = 0;
    virtual void on_forward_finished(int layer_id) = 0;
    virtual void on_backward_start(int layer_id) = 0;
    virtual void on_gradients_ready(int layer_id) = 0;

    template <typename T>
    friend class MultiSolver;
  };

  void add_callback(Callback* value) {
    root_solver_->add_callback(value);
    callbacks_.push_back(value);
  }

  int num_of_solvers();
  void pop_if_able();
  void add_another();

  virtual Dtype ForwardBackward();

  virtual void Solve();
  Net<Dtype>& net() {
    return *root_solver_->net();
  }

  const SolverParameter& param() const {
    return root_solver_->param();
  }

  shared_ptr<Solver<Dtype> > root_solver() {
    return root_solver_;
  }

 protected:
  shared_ptr<Solver<Dtype> > root_solver_;
  vector<shared_ptr<Solver<Dtype> > > worker_solvers;
  vector<Callback*> callbacks_;
  boost::recursive_mutex mtx;
};

}  // namespace caffe

#endif  // CAFFE_MULTISOLVER_HPP_

