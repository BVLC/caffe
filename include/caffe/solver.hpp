#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>
#include <memory>

#include "caffe/backend/device.hpp"
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
  * execution with a SIGINT (Ctrl-c).
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


class SolverBase {
 public:
  explicit SolverBase(Device* dev)
    : device_(dev), callbacks_(), requested_early_exit_(false) {
  }

  DataType data_type() const {
    return param_.data_type();
  }

  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
    friend class SolverBase;
  };

  virtual void Solve(const char* resume_file = NULL) = 0;
  virtual void Restore(const char* resume_file) = 0;
  virtual void Snapshot() = 0;
  virtual void SnapshotSolverState(const string& model_filename) = 0;

  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;

  void update_solver_param(const SolverParameter& param) {
    param_ = param;
  }
  inline const SolverParameter& param() const { return param_; }

  inline void Solve(const string resume_file) {
    Solve(resume_file.c_str());
  }

  virtual shared_ptr<NetBase> net_base() = 0;

  /**
   * @brief Returns the solver type.
   */
  virtual inline const char* type() const {
    return "";
  }

  inline Device *get_device() {
    return device_;
  }

  int_tp iter() {
    return iter_;
  }

  int_tp max_iter() {
    return param_.max_iter();
  }

  virtual const vector<shared_ptr<NetBase> > test_nets_bases() = 0;

  const vector<Callback*>& callbacks() const { return callbacks_; }

  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

 protected:
  SolverParameter param_;
  int_tp iter_;
  int_tp current_step_;

  // a function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  // Timing information, handy to tune e.g. nbr of GPUs
  Timer iteration_timer_;
  float iterations_last_;

  Device* device_;
  shared_ptr<DeviceProgram> device_program_;
  vector<Callback*> callbacks_;
};


/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template<typename Dtype>
class Solver : public SolverBase {
 public:
  explicit Solver(const SolverParameter& param, Device* dev);
  explicit Solver(const string& param_file, Device* dev);
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
  Dtype Step(int_tp iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  virtual void Restore(const char* resume_file);
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  virtual void Snapshot();
  virtual ~Solver() {}
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  virtual shared_ptr<NetBase> net_base() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  virtual const vector<shared_ptr<NetBase> > test_nets_bases() {
    std::vector<shared_ptr<NetBase> > converted_test_nets;
    for (size_t i = 0; i < test_nets_.size(); ++i) {
      converted_test_nets.push_back(std::static_pointer_cast<NetBase>(
          test_nets_[i]));
    }
    return converted_test_nets;
  }

  virtual void SnapshotSolverState(const string& model_filename) = 0;

  void CheckSnapshotWritePermissions();


  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;

 protected:
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  // The test routine
  void TestAll();
  void Test(const int_tp test_net_id = 0);
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void DisplayOutputBlobs(const int_tp net_id);
  void UpdateSmoothedLoss(Dtype loss, int_tp start_iter, int_tp average_loss);
  virtual void GenerateProgram() = 0;

  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  vector<Dtype> losses_;
  Dtype smoothed_loss_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};

}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
