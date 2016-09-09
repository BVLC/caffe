
#ifndef MULTI_NODE_WORKER_THREAD_H_
#define MULTI_NODE_WORKER_THREAD_H_


#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>

#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/multi_node/msg.hpp"
#include "caffe/multi_node/sk_server.hpp"
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
    omp_threads_ = -1;
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

  int SendMsg(shared_ptr<Msg> msg) {
    return sk_client_->SendMsg(msg);
  }

  void SetOMPThreads(int nthreads) {
    omp_threads_ = nthreads;
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
    boost::mutex::scoped_lock lock(new_solver_mutex_);
    const vector<Blob<Dtype>*>& root_params = proot->net()->learnable_params();

    Solver<Dtype> *new_solver = CreateSolver(proot, solver_param);
    if (new_solver == NULL) {
      return new_solver;
    }

    new_solver_cnt_++;

    const vector<Blob<Dtype>*>& new_params =
                                        new_solver->net()->learnable_params();
    /// share parameters with root solver
    for (int i = 0; i < new_params.size(); i++) {
      CHECK_EQ(new_params[i]->count(), root_params[i]->count());
      new_params[i]->ShareData(*root_params[i]);
    }

    LOG(INFO) << "created " << this->new_solver_cnt_ << " solvers";
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

 protected:
  // map layer id to its indices in learnable params
  vector<vector<int> > layer_id_to_params_;

  // get layer id from layer name
  unordered_map<string, int> layer_id_by_name_;

 protected:
  int worker_id_;
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

  // number of OpenMP threads it has
  int omp_threads_;

 protected:
  // serialize creating new solver within a process
  static boost::mutex  new_solver_mutex_;
  // number of solvers created
  static boost::atomic_int  new_solver_cnt_;

DISABLE_COPY_AND_ASSIGN(WorkerThread);
};

}  // end namespace caffe


#endif



