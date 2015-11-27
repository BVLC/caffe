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
class SolverContext : public Threaded {
 public:
  // Main solver does testing, display, snapshots etc.
  SolverContext(const Params<float>& params,
                const SolverParameter& solver_param, Solver<float>* solver)
      : params_(params),
        solver_param_(solver_param),
        worker_(solver == NULL),
        solver_(solver) {

    if (worker_) {
      solver_param_.clear_display();
      solver_param_.clear_snapshot();
    }
  }

  virtual void create_solver() {
    if (worker_) {
      solver_ = new SGDSolver<float>(solver_param_, true);
      CHECK(!solver_->test_nets().size());  // Only training
    }
  }

  virtual void delete_solver() {
    if (worker_)
      delete solver_;
  }

  inline Solver<float>* solver() const {
    return solver_;
  }

  virtual void stats(ostream& s) const {
  }

 protected:
  const Params<float>& params_;
  SolverParameter solver_param_;
  const bool worker_;
  Solver<float>* solver_;
};

// Runs a CPU solver on a thread
class CPUContext : public SolverContext {
 public:
  CPUContext(const Params<float>& params, const SolverParameter& solver_param,
             Solver<float>* solver = NULL)
      : SolverContext(params, solver_param, solver) {
  }

  void run() {
    create_solver();
    params_.configure(solver_);
    solver_->Solve();
    // Wait until asked to stop before destroying, monitor might
    // still be accessing fields
    if (worker_)
      while (!must_stop())
        sleep(1);
    delete_solver();
  }
};

#ifndef CPU_ONLY

// Runs a GPU solver on a thread
class GPUContext : public SolverContext {
 public:
  GPUContext(const Params<float>& params, const SolverParameter& solver_param,
               GPUParams<float>* gpu_params, Solver<float>* solver = NULL)
      : SolverContext(params, solver_param, solver),
        gpu_params_(gpu_params) {
  }

  void run() {
    create_solver();
    gpu_params_->configure(solver_);
    solver_->Solve();
    // Wait until asked to stop before destroying, monitor might
    // still be accessing fields
    if (worker_)
      while (!must_stop())
        sleep(1);
    delete_solver();
  }

 protected:
  GPUParams<float>* gpu_params_;
};

// Runs a GPU solver on a thread with CPU sync
class CPUGPUContext : public SolverContext {
 public:
  CPUGPUContext(const Params<float>& params,
                const SolverParameter& solver_param, Solver<float>* solver =
                NULL)
      : SolverContext(params, solver_param, solver),
        gpu_params_(),
        sync_() {
  }

  void run() {
    create_solver();
    gpu_params_ = new GPUParams<float>(params_, solver_param_.device_id());
    sync_ = new CPUGPUSync<float>(*gpu_params_);
    gpu_params_->configure(solver_);
    sync_->start();
    solver_->Solve();
    // Wait until asked to stop before destroying, monitor might
    // still be accessing fields
    if (worker_)
      while (!must_stop())
        sleep(1);
    delete sync_;
    delete gpu_params_;
    delete_solver();
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
  GPUParams<float>* gpu_params_;
  CPUGPUSync<float>* sync_;
};

#endif

// Displays stats about a set of solvers. Also keeps track and updates
// the global count of iterations (needed to adjust hyperparams).
class Monitor : public Threaded {
 public:
  Monitor(Params<float>& params, const vector<SolverContext*>& solvers)
      : params_(params),
        solvers_(solvers),
        total_iters_("total") {
  }

  virtual ~Monitor() {
  }

  void step(ostream* s = NULL) {
    *s << "Monitor - iters: ";

    int total = 0;
    bool all = true;  // TODO remove
    for (int i = 0; i < solvers_.size(); ++i) {
      SolverContext* ctx = solvers_[i];
      int n = ctx->solver() ? ctx->solver()->iter() : 0;
      total += n;
      if (s)
        *s << n << ", ";
      if (!n)
        all = false;
    }
    if (all) {
      //cudaProfilerStart();
      //LOG(INFO)<< "Started profiler\n";
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
    while (!must_stop()) {
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
