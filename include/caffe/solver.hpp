#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"

namespace caffe {

/**
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * a client may optionally provide in order to request early termination
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ComputeUpdateValue to compute a parameter update
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
  explicit Solver(const SolverParameter& param,
      const Solver* root_solver = NULL);
  explicit Solver(const string& param_file, const Solver* root_solver = NULL);
  void Init(const SolverParameter& param);
=======
  explicit Solver(const SolverParameter& param, bool skip_test_nets);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param, bool skip_test_nets);
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
  explicit Solver(const SolverParameter& param, bool skip_test_nets);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param, bool skip_test_nets);
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/device/blob.hpp
=======
  explicit Solver(const SolverParameter& param, bool skip_test_nets);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param, bool skip_test_nets);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
  explicit Solver(const SolverParameter& param, bool skip_test_nets);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param, bool skip_test_nets);
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
  explicit Solver(const SolverParameter& param, bool skip_test_nets);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param, bool skip_test_nets);
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
  explicit Solver(const SolverParameter& param, bool skip_test_nets);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param, bool skip_test_nets);
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
  explicit Solver(const SolverParameter& param, bool skip_test_nets);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param, bool skip_test_nets);
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
  explicit Solver(const SolverParameter& param, bool skip_test_nets);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param, bool skip_test_nets);
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  inline int iter() { return iter_; }
  inline int *iter_total() { return iter_total_; }
  inline void iter_total(int *value) { iter_total_ = value; }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
  int iter() { return iter_; }
>>>>>>> pod-caffe-pod.hpp-merge

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
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
>>>>>>> caffe
  int iter() { return iter_; }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
  int iter() { return iter_; }
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> caffe
  int iter() { return iter_; }
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  int iter() { return iter_; }
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
  int iter() { return iter_; }
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
  int iter() { return iter_; }
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
  int iter() { return iter_; }
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/common.hpp

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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
  inline int iter() { return iter_; }
  inline int *iter_total() { return iter_total_; }
  inline void iter_total(int *value) { iter_total_ = value; }
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  inline int iter() { return iter_; }
  inline int *iter_total() { return iter_total_; }
  inline void iter_total(int *value) { iter_total_ = value; }
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp
  inline int iter() { return iter_; }
  inline int *iter_total() { return iter_total_; }
  inline void iter_total(int *value) { iter_total_ = value; }
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp

 protected:
  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
=======
=======
>>>>>>> pod/device/blob.hpp
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
=======
<<<<<<< HEAD
<<<<<<< HEAD
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
=======
  virtual void SnapshotSolverState(SolverState* state) = 0;
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual void RestoreSolverState(const SolverState& state) = 0;
>>>>>>> origin/BVLC/parallel
=======
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  void DisplayOutputBlobs(const int net_id);

  SolverParameter param_;
  int iter_;
<<<<<<< HEAD
  // Points to iter_ by default, but can be overriden, e.g. to a global
  // counter if multiple solvers contribute iterations to the same model.
  int *iter_total_;
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
  // Points to iter_ by default, but can be overriden, e.g. to a global
  // counter if multiple solvers contribute iterations to the same model.
  int *iter_total_;
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  vector<Callback*> callbacks_;
<<<<<<< HEAD
=======

  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  const Solver* const root_solver_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;
>>>>>>> pod-caffe-pod.hpp-merge

<<<<<<< HEAD
  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  const Solver* const root_solver_;
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
  virtual void SnapshotSolverState(SolverState* state) = 0;
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual void RestoreSolverState(const SolverState& state) = 0;
>>>>>>> origin/BVLC/parallel
=======
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
=======
  virtual void SnapshotSolverState(SolverState* state) = 0;
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual void RestoreSolverState(const SolverState& state) = 0;
>>>>>>> origin/BVLC/parallel
=======
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
=======
<<<<<<< HEAD
<<<<<<< HEAD
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
=======
  virtual void SnapshotSolverState(SolverState* state) = 0;
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual void RestoreSolverState(const SolverState& state) = 0;
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/device/blob.hpp
=======
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
<<<<<<< HEAD
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
  void DisplayOutputBlobs(const int net_id);

  SolverParameter param_;
  int iter_;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
  // Points to iter_ by default, but can be overriden, e.g. to a global
  // counter if multiple solvers contribute iterations to the same model.
  int *iter_total_;
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  // Points to iter_ by default, but can be overriden, e.g. to a global
  // counter if multiple solvers contribute iterations to the same model.
  int *iter_total_;
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  // Points to iter_ by default, but can be overriden, e.g. to a global
  // counter if multiple solvers contribute iterations to the same model.
  int *iter_total_;
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  // Points to iter_ by default, but can be overriden, e.g. to a global
  // counter if multiple solvers contribute iterations to the same model.
  int *iter_total_;
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
  // Points to iter_ by default, but can be overriden, e.g. to a global
  // counter if multiple solvers contribute iterations to the same model.
  int *iter_total_;
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  vector<Callback*> callbacks_;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp

  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  const Solver* const root_solver_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======

  // True iff a request to stop early was received.
  bool requested_early_exit_;
>>>>>>> device-abstraction
=======

  // True iff a request to stop early was received.
  bool requested_early_exit_;
>>>>>>> pod-caffe-pod.hpp-merge

  // True iff a request to stop early was received.
  bool requested_early_exit_;
>>>>>>> pod/caffe-merge

  // True iff a request to stop early was received.
  bool requested_early_exit_;
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======

  // True iff a request to stop early was received.
  bool requested_early_exit_;
>>>>>>> pod/caffe-merge
=======

  // True iff a request to stop early was received.
  bool requested_early_exit_;
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp

  // True iff a request to stop early was received.
  bool requested_early_exit_;
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp

<<<<<<< HEAD
  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  const Solver* const root_solver_;
=======
  DISABLE_COPY_AND_ASSIGN(Solver);
};

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
/**
 * @brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
=======

/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> origin/BVLC/parallel
=======
/**
 * @brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
/**
 * @brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
  DISABLE_COPY_AND_ASSIGN(Solver);
};

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
/**
 * @brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
=======

/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
/**
 * @brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
 */
template <typename Dtype>
class WorkerSolver : public Solver<Dtype> {
 public:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
  explicit SGDSolver(const SolverParameter& param, bool skip_test_nets = false)
      : Solver<Dtype>(param, skip_test_nets) {}
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) {}
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }
<<<<<<< HEAD
=======

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }
>>>>>>> pod-caffe-pod.hpp-merge
=======

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }
>>>>>>> pod/caffe-merge
=======

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }
>>>>>>> pod-caffe-pod.hpp-merge

 protected:
<<<<<<< HEAD
=======
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp

<<<<<<< HEAD
  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
<<<<<<< HEAD
=======
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}
=======
>>>>>>> pod/device/blob.hpp

<<<<<<< HEAD
  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
<<<<<<< HEAD
=======
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}
=======
>>>>>>> pod/device/blob.hpp

<<<<<<< HEAD
  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
<<<<<<< HEAD
=======
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}
=======
>>>>>>> pod/device/blob.hpp

<<<<<<< HEAD
  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
<<<<<<< HEAD
=======
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}

<<<<<<< HEAD
  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }
>>>>>>> origin/BVLC/parallel

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;
=======
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}
>>>>>>> device-abstraction

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
=======
 protected:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======

 protected:
>>>>>>> pod-caffe-pod.hpp-merge
=======

 protected:
>>>>>>> pod/caffe-merge
=======

 protected:
<<<<<<< HEAD
=======
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}

 protected:
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> caffe
=======
>>>>>>> device-abstraction
=======

 protected:
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======

 protected:
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
=======

 protected:
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
};

=======
  virtual void PreSolve();
  Dtype GetLearningRate();
  virtual void ComputeUpdateValue();
  virtual void SnapshotSolverState(SolverState * state);
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

>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
};

>>>>>>> caffe
=======
};

>>>>>>> device-abstraction
=======
};

>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
};

>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
};

>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
=======
};

>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
};

>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
