#ifndef CAFFE_SOLVER_TRACE_HPP_
#define CAFFE_SOLVER_TRACE_HPP_
#include <string>
#include <vector>

#include "caffe/solver.hpp"

namespace caffe {

template <typename Dtype> class Solver;

/**
 * @brief Maintains a history of the Solver process so that the learning can
 *        be analyzied
 * 
 * The solver trace stores the values over time of certain aspects of learning
 * such as the weights, biases, activations, gradients and test and training
 * errors. These are stored and then from time to time saved to disk.
 * 
 * @note Since every run has only one solver trace, and it continually extended
 * with new values there is no real disadvantage to updating and saving often,
 * only the memory used and the time writing to disk.
 */
template <typename Dtype>
class SolverTrace {
 public:
  explicit SolverTrace(const SolverParameter& param,
                       const Solver<Dtype>* solver);
  virtual ~SolverTrace() {}
  /** @brief Get a const reference to the history */
  virtual const TraceDigest& get_digest() const;
  virtual void update_trace_train(SolverAction::Enum request);
  virtual void update_trace_test_loss(int test_net_id, Dtype loss);
  virtual void update_trace_test_score(int test_net_id,
                                       const string& output_name,
                                       Dtype loss_weight,
                                       Dtype mean_score);
  /** @brief Save the history of the solver to a proto file */
  virtual void Save() const;
 protected:
  SolverParameter param_;
  SolverTraceParameter trace_param_;
  shared_ptr<TraceDigest> trace_digest_;       /// History of the solver
  const Solver<Dtype>* solver_;                /// Solver to get history for
  int save_trace_;                             /// Interval when to save trace
  DISABLE_COPY_AND_ASSIGN(SolverTrace);
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_TRACE_HPP_
