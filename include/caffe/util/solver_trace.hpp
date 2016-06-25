#ifndef CAFFE_SOLVER_TRACE_HPP_
#define CAFFE_SOLVER_TRACE_HPP_
#include <map>
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
  const TraceDigest& get_digest() const;
  /** @brief Add trace of state the solver is in during training to digest */
  void update_trace_train(SolverAction::Enum request);
  void update_trace_train_loss(Dtype loss,
                                       Dtype smoothed_loss);
  void update_trace_test_loss(int test_net_id, Dtype loss);
  void update_trace_test_score(int test_net_id,
                                       const string& output_name,
                                       Dtype loss_weight,
                                       Dtype mean_score);
  /** @brief Save the history of the solver to a proto file */
  void Save() const;
  /** @brief Restore the history of the solver from a proto file */
  void Restore(const string& trace_filename);

 protected:
  void update_weight_trace();                  /// Add weight history to digest
  void update_activation_trace();
  void update_diff_trace();
  void blob_trace(int trace_interval, int num_traces, bool use_data);
  void init_weight_trace();
  void init_activation_trace();
  void init_diff_trace();
  SolverParameter param_;
  SolverTraceParameter trace_param_;
  shared_ptr<TraceDigest> trace_digest_;       /// History of the solver
  const Solver<Dtype>* solver_;                /// Solver to get history for
  int save_trace_;                             /// Interval when to save trace
  int start_iter_;                             /// Iter where the solver starts
  int last_iter_;                              /// The last iter we updated
  string filename_;                            /// File for the solver trace
  int num_activation_traces_;
  int activation_trace_interval_;
  int num_weight_traces_;
  int weight_trace_interval_;
  int num_diff_traces_;
  int diff_trace_interval_;
  std::map<string, int> activation_name_to_index_;
  DISABLE_COPY_AND_ASSIGN(SolverTrace);
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_TRACE_HPP_
