#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"

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
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param,
      const Solver* root_solver = NULL);
  explicit Solver(const string& param_file, const Solver* root_solver = NULL);
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  void Restore(const char* resume_file);
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  virtual ~Solver() {}
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() { return iter_; }

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
  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void DisplayOutputBlobs(const int net_id);
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

  SolverParameter param_;
  int iter_;
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  vector<Callback*> callbacks_;
  vector<Dtype> losses_;
  Dtype smoothed_loss_;

  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  const Solver* const root_solver_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};

/**
 * @brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
 */
template <typename Dtype>
class WorkerSolver : public Solver<Dtype> {
 public:
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}

 protected:
  void ApplyUpdate() {}
  void SnapshotSolverState(const string& model_filename) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromBinaryProto(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromHDF5(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
