
#ifndef MULTI_NODE_WORKER_THREAD_H_
#define MULTI_NODE_WORKER_THREAD_H_


#include "caffe/multi_node/msg.hpp"
#include "caffe/multi_node/sk_server.hpp"
#include "caffe/caffe.hpp"

#include <boost/thread.hpp>
#include <boost/atomic.hpp>

namespace caffe {
/**
* Base class of the convolution, FC, PS threads
* Set up the communication enviroment with routing thread
*/
template <typename Dtype>
class WorkerThread : public InternalThread
{
public:
  WorkerThread() {
    worker_id_ = -1;
    queue_size_ = 0;
  }

  virtual ~WorkerThread() {
    StopInternalThread();
  }

  void SetWorkerId(int id) { worker_id_ = id; }
  int GetWorkerId() { return worker_id_; }

  void SetAddr(const string& addr) { addr_ = addr; }
  
  int SendMsg(shared_ptr<Msg> msg) {
    return sk_client_->SendMsg(msg);
  }

  shared_ptr<Msg> RecvMsg(bool blocked) {
    shared_ptr<Msg> m = sk_client_->RecvMsg(blocked);
    Dequeue();
    return m;
  }

  virtual void InternalThreadEntry() {
    sk_client_.reset(new SkSock(ZMQ_PAIR));
    sk_client_->Connect(addr_);

    Run();
  }
  
  /// should be reimplented in the working threads
  virtual void Run() = 0;
  
  inline void Enqueue() { queue_size_++; }
  inline int QueueSize() { return queue_size_; }

protected:
  inline void Dequeue() { queue_size_--; }
  
  virtual Solver<Dtype> *NewSolver(Solver<Dtype> *proot, const SolverParameter& solver_param) {
    boost::mutex::scoped_lock lock(new_solver_mutex_);
    const vector<Blob<Dtype>*>& root_params = proot->net()->learnable_params();

    Solver<Dtype> *new_solver = CreateSolver(proot, solver_param);
    if (new_solver == NULL) {
      return new_solver;
    }

    new_solver_cnt_++;
    
    const vector<Blob<Dtype>*>& new_params = new_solver->net()->learnable_params();
    /// share parameters with root solver
    for (int i = 0; i < new_params.size(); i++) {
      CHECK_EQ(new_params[i]->count(), root_params[i]->count());
      new_params[i]->ShareData(*root_params[i]);
    }

    return new_solver;
  }

  virtual Solver<Dtype> *CreateSolver(const Solver<Dtype> *root_solver, const SolverParameter& solver_param) = 0;

protected:
  int worker_id_;
  /// in-process communication addr with routing thread
  string addr_;
  shared_ptr<SkSock> sk_client_;

  // rough count of the queue's length
  boost::atomic_int queue_size_;

protected:
  // serialize creating new solver within a process
  static boost::mutex       new_solver_mutex_;
  // number of solvers created
  static boost::atomic_int  new_solver_cnt_;

DISABLE_COPY_AND_ASSIGN(WorkerThread);
};


} // end caffe


#endif



