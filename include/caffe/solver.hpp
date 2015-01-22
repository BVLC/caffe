#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <boost/circular_buffer.hpp>

#include <string>
#include <vector>

#include "caffe/net.hpp"

namespace caffe {

/**
 * @brief The SolverResult struct holds output of each stepping of Solver
 */
template <typename Dtype>
struct SolverResult {
  string blob_name;
  Dtype loss_weight;
  vector<Dtype> blob_data;
};

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ComputeUpdateValue to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);

  /**
   * @brief Next
   * @param output The output blobs before the network is updated
   * @return The total loss of this round of forwarding
   */
  Dtype Next(vector<shared_ptr<SolverResult<Dtype> > >* output = 0);
  virtual ~Solver() {}
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline shared_ptr<const Net<Dtype> > net() const { return net_; }

  int iter() const { return iter_; }
  int current_step() const { return current_step_; }
  const SolverParameter& param() const { return param_; }
  virtual void SnapshotSolverState(SolverState* state) const = 0;
  virtual void RestoreSolverState(const SolverState& state) = 0;

  // Snapshot to default filenames
  void Snapshot() const;
  void Snapshot(const string& model_filename,
                const string& snapshot_filename) const;

  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);

  typedef boost::circular_buffer<Dtype> loss_container_type;
  const loss_container_type& last_losses() const { return last_losses_; }

  Dtype smoothed_loss() const { return smoothed_loss_; }

 protected:
  // Get the update value for the current iteration.
  virtual void ComputeUpdateValue() = 0;

  SolverParameter param_;
  int iter_;
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  loss_container_type last_losses_;
  Dtype smoothed_loss_;

 private:
  void InitTrainNet();
  void Init(const SolverParameter& param);

  DISABLE_COPY_AND_ASSIGN(Solver);
};

/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
  void PreSolve();
  Dtype GetLearningRate();
  virtual void ComputeUpdateValue();
  virtual void SnapshotSolverState(SolverState * state) const;
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) {}
  explicit NesterovSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {}

 protected:
  virtual void ComputeUpdateValue();

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }

 protected:
  virtual void ComputeUpdateValue();
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};

template <typename Dtype>
Solver<Dtype>* GetSolver(const SolverParameter& param) {
  SolverParameter_SolverType type = param.solver_type();

  switch (type) {
  case SolverParameter_SolverType_SGD:
      return new SGDSolver<Dtype>(param);
  case SolverParameter_SolverType_NESTEROV:
      return new NesterovSolver<Dtype>(param);
  case SolverParameter_SolverType_ADAGRAD:
      return new AdaGradSolver<Dtype>(param);
  default:
      LOG(FATAL) << "Unknown SolverType: " << type;
  }
  return (Solver<Dtype>*) NULL;
}

}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
