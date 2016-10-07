
#ifndef MULTI_NODE_WORKER_THREAD_H_
#define MULTI_NODE_WORKER_THREAD_H_

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>

#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/multi_node/msg.hpp"
#include "caffe/multi_node/param_helper.hpp"
#include "caffe/multi_node/sk_server.hpp"
#include "caffe/multi_node/solver_pool.hpp"
#include "caffe/sgd_solvers.hpp"


using boost::unordered_map;

namespace caffe {
/**
* Base class of the convolution, FC, PS threads
* Set up the communication enviroment with routing thread
*/
template <typename Dtype>
class WorkerThread : public InternalThread {
 public:
  WorkerThread() {
    worker_id_ = -1;
    queue_size_ = 0;
    num_workers_ = 0;
    cpu_socket_idx_ = 0;
    num_clients_ = 0;
  }

  virtual ~WorkerThread() {
    StopInternalThread();
  }

  void SetWorkerId(int id) { worker_id_ = id; }
  int GetWorkerId() { return worker_id_; }

  void SetWorkerNum(int nworkers) { num_workers_ = nworkers; }
  int GetWorkerNum() { return num_workers_; }

  void SetClientAddr(const string& addr) { client_addr_ = addr; }
  void SetPriorAddr(const string& addr) { prior_addr_ = addr; }

  void SetOMPCores(const vector<int> omp_cores) {
    omp_cores_ = omp_cores;

    if (omp_cores_.size() > 0) {
      cpu_socket_idx_ = NodeEnv::Instance()->GetSocketIndex(omp_cores_[0]);
    }
  }

  inline int GetClients() { return num_clients_; }

  inline void AddClient() { num_clients_++; }

  int SendMsg(shared_ptr<Msg> msg) {
    return sk_client_->SendMsg(msg);
  }

  shared_ptr<Msg> RecvMsg(bool blocked) {
    shared_ptr<Msg> m;

    if (blocked) {
      zmq_poll(poll_table_, NUM_SUB_SOCKS, -1);
    } else {
      zmq_poll(poll_table_, NUM_SUB_SOCKS, 0);
    }

    if (poll_table_[0].revents & ZMQ_POLLIN) {
      m = sk_prior_->RecvMsg(true);
      Dequeue();
      return m;
    }

    if (poll_table_[1].revents & ZMQ_POLLIN) {
      m = sk_client_->RecvMsg(true);
      Dequeue();
      return m;
    }

    return m;
  }

  virtual void InternalThreadEntry() {
    sk_client_.reset(new SkSock(ZMQ_PAIR));
    sk_client_->Connect(client_addr_);

    sk_prior_.reset(new SkSock(ZMQ_PAIR));
    sk_prior_->Connect(prior_addr_);

    // prioritize sockets with smaller indices
    poll_table_[0].socket = sk_prior_->GetSock();
    poll_table_[1].socket = sk_client_->GetSock();

    for (int i = 0; i < NUM_SUB_SOCKS; i++) {
      poll_table_[i].events = ZMQ_POLLIN;
      poll_table_[i].fd = 0;
      poll_table_[i].revents = 0;
    }

    Run();
  }

  /// should be reimplented in the working threads
  virtual void Run() = 0;

  inline void Enqueue() { queue_size_++; }

  inline int QueueSize() { return queue_size_; }

 public:
  static void InitParamSolver(Solver<Dtype> *proot, int nsockets) {
    param_solvers_.resize(nsockets);
    param_solvers_[0] = proot;
    for (int i = 1; i < nsockets; i++) {
      param_solvers_[i] = NULL;
    }
  }

 protected:
  inline void Dequeue() { queue_size_--; }

  // init param map: map from layer_id to learnable_id
  int InitParamMap(shared_ptr<Net<Dtype> > net);

  // check whether the layer is learnable
  bool IsLearnable(int layer_id) {
    if (layer_id >= layer_id_to_params_.size()) {
      return false;
    }

    return layer_id_to_params_[layer_id].size() > 0;
  }

  // send exit message to worker threads
  void SendExit();

  // Bind main thread a specific CPU core
  void BindCore(int core_id);

  // Bind OMP threads to a list of cores
  void BindOMPThreads(const vector<int>& core_list);

  const vector<int>& GetLearnableIndices(int layer_id) {
    CHECK_GT(layer_id_to_params_.size(), layer_id);

    return layer_id_to_params_[layer_id];
  }

  int GetLayerId(const string& layer_name) {
    unordered_map<string, int>::iterator iter = layer_id_by_name_.end();
    iter = layer_id_by_name_.find(layer_name);

    CHECK(iter != layer_id_by_name_.end());

    return iter->second;
  }

  Solver<Dtype> *NewSolver(Solver<Dtype> *proot,
                           const SolverParameter& solver_param) {
    Solver<Dtype> *new_solver = CreateSolver(proot, solver_param);
    if (new_solver == NULL) {
      return new_solver;
    }

    boost::mutex::scoped_lock lock(new_solver_mutex_);
    new_solver_cnt_++;

    LOG(INFO) << "created " << this->new_solver_cnt_ << " solvers"
              << ", at socket: " << cpu_socket_idx_;

    const vector<Blob<Dtype>*> *proot_params = NULL;

    if (param_solvers_.size() > 0 && param_solvers_[cpu_socket_idx_] == NULL) {
      // use the first solver as param solver of the socket
      param_solvers_[cpu_socket_idx_] = new_solver;
      return new_solver;
    } else if (param_solvers_.size() <= 0) {
      proot_params = &proot->net()->learnable_params();
    } else {
      proot_params = &param_solvers_[cpu_socket_idx_]->net()->learnable_params();
    }

    const vector<Blob<Dtype>*>& new_params =
                                        new_solver->net()->learnable_params();
    /// share parameters with root solver
    for (int i = 0; i < new_params.size(); i++) {
      CHECK_EQ(new_params[i]->count(), proot_params->at(i)->count());
      new_params[i]->ShareData(*(proot_params->at(i)));
    }

    return new_solver;
  }

  virtual Solver<Dtype> *CreateSolver(const Solver<Dtype> *root_solver,
                                      const SolverParameter& solver_param) {
    Caffe::set_root_solver(false);
    SGDSolver<Dtype> *proot = (SGDSolver<Dtype> *)root_solver;
    WorkerSolver<Dtype> *solver = new WorkerSolver<Dtype>(solver_param, proot);

    // Disable bottom backward to data layer.
    DisableBottomBwdToData(solver);

    return solver;
  }

  // disable bottom backward to data layer.
  void DisableBottomBwdToData(WorkerSolver<Dtype> * solver) {
    vector<Blob<Dtype>*> data_top_blobs;
    shared_ptr<Net<Dtype> > net = solver->net();
    const vector<shared_ptr<Layer<Dtype> > >& layers = net->layers();

    for (int layer_id = 0; layer_id < layers.size(); layer_id ++) {
      if (net->bottom_vecs()[layer_id].size() == 0) {
        const vector<Blob<Dtype>*> & top_blobs = net->top_vecs()[layer_id];
        data_top_blobs.insert(data_top_blobs.end(),
                              top_blobs.begin(),
                              top_blobs.end());
      }
    }

    vector<vector<bool> >& bottom_need_bwd =
      const_cast<vector<vector<bool> >&>(net->bottom_need_backward());

    for (int layer_id = 0; layer_id < layers.size(); layer_id ++) {
      const vector<Blob<Dtype>*> & bottom_blobs = net->bottom_vecs()[layer_id];

      for (int i = 0; i < bottom_blobs.size(); i ++) {
        if (std::find(data_top_blobs.begin(),
                     data_top_blobs.end(),
                     bottom_blobs[i]) != data_top_blobs.end()) {
          CHECK(i < bottom_need_bwd[layer_id].size());
          bottom_need_bwd[layer_id][i] = false;
        }
      }
    }
  }

  inline void PushFreeSolver(Solver<Dtype> *p) {
    solver_pool_.PushFreeSolver(p);
  }

  inline Solver<Dtype> *PopFreeSolver() {
    return solver_pool_.PopFreeSolver();
  }

  inline Solver<Dtype> *FindSolver(int64_t msg_id) {
    return solver_pool_.FindSolver(msg_id);
  }

  inline void BindSolver(Solver<Dtype> *psolver, int64_t msg_id) {
    solver_pool_.BindSolver(psolver, msg_id);
  }

  inline void ReleaseSolver(int64_t msg_id) {
    solver_pool_.ReleaseSolver(msg_id);
  }

  inline void RemoveBind(int64_t msg_id) {
    solver_pool_.RemoveBind(msg_id);
  }

 protected:
  static void UpdateSocketParams() {
    if (param_solvers_.size() <= 0) {
      return;
    }

    Solver<Dtype> *proot = param_solvers_[0];
    for (int i = 1; i < param_solvers_.size(); i++) {
      if (param_solvers_[i] != NULL) {
        ParamHelper<Dtype>::CopyDataFromNet(param_solvers_[i]->net(),
                                            proot->net());
      }
    }
  }

 protected:
  // map layer id to its indices in learnable params
  vector<vector<int> > layer_id_to_params_;

  // get layer id from layer name
  unordered_map<string, int> layer_id_by_name_;

 protected:
  int worker_id_;

  // total number of worker threads the node started
  int num_workers_;

  /// in-process communication addr with routing thread
  string client_addr_;

  shared_ptr<SkSock> sk_client_;

  // for priority queue
  string prior_addr_;
  shared_ptr<SkSock> sk_prior_;

  // number of socks the thread has
  static const int NUM_SUB_SOCKS = 2;

  // polling table
  zmq_pollitem_t poll_table_[NUM_SUB_SOCKS];

  // rough count of the queue's length
  boost::atomic_int queue_size_;

  // solver pool to store free solvers
  SolverPool<Dtype> solver_pool_;

  // cores used by openmp threads
  vector<int> omp_cores_;

  // socket index the omp threads works on
  int cpu_socket_idx_;

  // number of clients it serves
  boost::atomic_int num_clients_;

 protected:
  // serialize creating new solver within a process
  static boost::mutex  new_solver_mutex_;

  // number of solvers created
  static boost::atomic_int  new_solver_cnt_;

  // each socket has a different parameter
  static vector<Solver<Dtype> *> param_solvers_;

DISABLE_COPY_AND_ASSIGN(WorkerThread);
};

}  // end namespace caffe


#endif



