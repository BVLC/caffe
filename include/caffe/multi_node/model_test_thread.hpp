
#ifndef MULTI_NODE_TEST_THREAD_H_
#define MULTI_NODE_TEST_THREAD_H_

#include <map>
#include <vector>

#include "caffe/multi_node/node_env.hpp"
#include "caffe/multi_node/param_helper.hpp"
#include "caffe/multi_node/worker_thread.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
class TestThread : public WorkerThread<Dtype> {
 public:
  TestThread() {
    ps_ids_ = NodeEnv::Instance()->ps_ids();
    fc_ids_ = NodeEnv::Instance()->fc_ids();

    ResetUpdateMap();
    train_iter_ = 0;
    tested_iter_ = 0;
    snapshot_iter_ = 0;
    param_ = NodeEnv::Instance()->SolverParam();
    solver_ = new SGDSolver<Dtype>(param_);
    max_iter_ = NodeEnv::Instance()->SolverParam().max_iter();
    need_exit_ = false;
  }

  virtual ~TestThread() { delete solver_; }

  virtual void Run();

 protected:
  void UpdateTrainIter(shared_ptr<Msg> m);

  void ResetUpdateMap() {
    for (int i = 0; i < ps_ids_.size(); i++) {
      updated_map_[ps_ids_[i]] = false;
    }
    for (int i = 0; i < fc_ids_.size(); i++) {
      updated_map_[fc_ids_[i]] = false;
    }
  }

  void SendParamRquest();

  int UpdateParam(shared_ptr<Msg> m);

  virtual Solver<Dtype> *CreateSolver(const Solver<Dtype> *root_solver,
                                      const SolverParameter& solver_param) {
    return NULL;
  }

 protected:
  vector<int> ps_ids_;
  vector<int> fc_ids_;

  map<int, bool> updated_map_;

  // number of traning iterations
  int train_iter_;

  // the iteration which has been tested
  int tested_iter_;

  // the iter which has a snapshot
  int snapshot_iter_;

  SolverParameter param_;

  SGDSolver<Dtype> *solver_;

  int max_iter_;

  bool need_exit_;
};

}  // end namespace caffe

#endif


