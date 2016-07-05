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
    virtual void on_start() = 0;  // from Solver<Dtype>::Callback
    virtual void on_gradients_ready() = 0;  // from Solver<Dtype>::Callback

    template <typename T>
    friend class MultiSolver;
  };

  void add_callback(Callback* value) {
    root_solver_->add_callback(value);
    callbacks_.push_back(value);
  }

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

 private:
  virtual Dtype ForwardBackwardImpl(bool first, bool last);

 protected:
  shared_ptr<Solver<Dtype> > root_solver_;
  int iter_size;
  vector<Callback*> callbacks_;
};

}  // namespace caffe

#endif  // CAFFE_MULTISOLVER_HPP_

