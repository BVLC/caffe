#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

/**
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
  namespace SolverAction {
    enum Enum {
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief Type of a function that handles custom snapshotting.
 */
typedef boost::function<void()> SnapshotCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function, simply dispatched to a protected
  // method. By default, iter will be zero. Pass in a non-zero iter number to
  // resume training for a pre-trained net.
  inline void Solve(const char* resume_file = NULL) {
    Solve(resume_file, NULL, NULL);
  }
  inline void Solve(const string& resume_file) { Solve(resume_file.c_str()); }
  inline void Solve(const NetParameter& net_param, const SolverState& state) {
    Solve(NULL, &net_param, &state);
  }
  void Step(int iters);
  // The file version of the Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods to restore the state from the
  // appropriate snapshot type. The protocol buffer version of the Restore
  // method restores the learned net and dispatches to the RestoreSolverState
  // protected method to restore additional state information. You must
  // implement all of these protected dispatch methods.
  void Restore(const char* resume_file);
  void Restore(const NetParameter& net_param, const SolverState& state);
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You must implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk or handled by a callback together with the learned net.
  // However, it is valid to implement custom snapshotting without SolverState.
  void Snapshot();
  void Snapshot(NetParameter* net_param, SolverState* state = NULL);
  // Client of the Solver optionally may call this in order to set a function
  // that replaces or extends standard snapshotting.
  void SetSnapshotFunction(SnapshotCallback func);
  virtual ~Solver() {}
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net<Dtype> > net() const { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() const {
    return test_nets_;
  }
  int iter() const { return iter_; }
  const vector<int>& mean_scores_output_ids(const int test_net_id) const {
    return mean_scores_output_ids_[test_net_id];
  }
  const vector<Dtype>& mean_scores(const int test_net_id) const {
    return mean_scores_[test_net_id];
  }
  Dtype smoothed_loss() const { return smoothed_loss_; }
  Dtype learning_rate() const { return learning_rate_; }

  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

  void CheckSnapshotWritePermissions();
  /**
   * @brief Returns the solver type.
   */
  virtual inline const char* type() const { return ""; }

 protected:
  virtual void Solve(const char* resume_file,
                     const NetParameter* net_param, const SolverState* state);
  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;
  string SnapshotFilename(const string& extension) const;
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void SnapshotSolverState(SolverState* state) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  virtual void RestoreSolverState(const SolverState& state) = 0;
  void DisplayOutputBlobs(const int net_id);
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

  SolverParameter param_;
  int iter_;
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  vector<Callback*> callbacks_;
  vector<vector<int> > mean_scores_output_ids_;
  vector<vector<Dtype> > mean_scores_;
  vector<Dtype> losses_;
  Dtype smoothed_loss_;
  Dtype learning_rate_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // A function that can be set by a client of the Solver to bypass standard
  // snapshotting and provide custom functionality. For example, you may call
  // the protocol buffer version of the Snapshot method from this function.
  SnapshotCallback custom_snapshot_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  // Timing information, handy to tune e.g. nbr of GPUs
  Timer iteration_timer_;
  float iterations_last_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
