#include <caffe/parallel.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/solver.hpp>
#include <glog/logging.h>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace caffe;

// Shared code for parallel examples. Should be replaced by some kind of cluster
// deployment and visualization solution.

// Context for a solver running in a thread. Both initialization and run
// of the solver are done on the thread, to point to the same instance of the
// thread-local Caffe singleton.
class SolverContext: public PThread {
public:
  // Main solver does testing, display, snapshots etc.
  SolverContext(Params<float>& params, Solver<float>& solver) :
      params_(params), solver_param_(solver.param()), index_(0), //
      solver_(&solver) {

    // Main solver runs on current thread
    init_as_current();
  }

  // Other solvers do only training and run in separate threads
  SolverContext(Params<float>& params, const SolverParameter& solver_param,
      int index) :
      params_(params), solver_param_(solver_param), index_(index), solver_() {

    solver_param_.clear_display();
    solver_param_.clear_snapshot();
  }

  virtual void create_solver() {
    if (index_) {
      solver_ = new SGDSolver<float>(solver_param_);
      solver_->test_nets().clear(); // Only training
    }
  }

  inline Solver<float>* solver() const {
    return solver_;
  }

  virtual void stats(ostream& s) const {
  }

protected:
  Params<float>& params_;
  SolverParameter solver_param_;
  const int index_;
  Solver<float>* solver_; // TODO delete in thread
};

// Runs a CPU solver on a thread
class CPUContext: public SolverContext {
public:
  CPUContext(Params<float>& params, Solver<float>& solver) :
      SolverContext(params, solver) {
  }

  CPUContext(Params<float>& params, const SolverParameter& solver_param,
      int index) :
      SolverContext(params, solver_param, index) {
  }

  void run() {
    create_solver();
    CPUSync<float> sync(params_, *solver_);
    solver_->Solve();
  }
};

// Runs a GPU solver on a thread
class GPUContext: public SolverContext {
public:
  GPUContext(Params<float>& params, Solver<float>& solver) :
      SolverContext(params, solver), sync_() {
  }

  GPUContext(Params<float>& params, const SolverParameter& solver_param,
      int index) :
      SolverContext(params, solver_param, index), sync_() {
  }

  inline GPUSync<float>* sync() const {
    return sync_;
  }

  void run() {
    create_solver();
    sync_ = new GPUSync<float>(params_, *solver_);
    sync_->start();
    solver_->Solve();
  }

  virtual void stats(ostream& s) const {
    s << "GPU " << solver_param_.device_id() << " ";
    if (sync_) {
      sync_->calls().show(s);
      s << ", ";
      sync_->cycles().show(s);
    } else
      s << "starting";
    s << ", ";
  }

protected:
  GPUSync<float>* sync_;
};

// Displays stats about a set of solvers. Also keeps track and updates
// the global count of iterations (needed to adjust hyperparams).
class Monitor: public PThread {
public:
  Monitor(Params<float>& params, const vector<SolverContext*>& solvers) :
      params_(params), solvers_(solvers), total_iters_("total") {
  }

  virtual ~Monitor() {
  }

  void step(ostream* s = NULL) {
    *s << "Monitor - iters: ";

    int total = 0;
    for (int i = 0; i < solvers_.size(); ++i) {
      SolverContext* ctx = solvers_[i];
      int n = ctx->solver() ? ctx->solver()->iter() : 0;
      total += n;
      if (s)
        *s << n << ", ";
    }
    params_.iterations(total);
    total_iters_.value(total);
    if (s) {
      total_iters_.show(*s);
      *s << ", ";
      for (int i = 0; i < solvers_.size(); ++i)
        solvers_[i]->stats(*s);
    }
  }

  void run() {
    int every_seconds = 10;
    time_t start = time(0);
    for (;;) {
      sleep(every_seconds);

      ostringstream s;
      step(&s);
      s << "\n";
      LOG(INFO)<< s.str();
      LOG(INFO)<< "Training time: " << (time(0) - start);
    }
  }

protected:
  Params<float>& params_;
  const vector<SolverContext*>& solvers_;
  Meter total_iters_;
};
