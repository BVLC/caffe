
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

  void SetClientAddr(const string& addr) { client_addr_ = addr; }
  void SetPriorAddr(const string& addr) { prior_addr_ = addr; }
  
  int SendMsg(shared_ptr<Msg> msg) {
    return sk_client_->SendMsg(msg);
  }

  shared_ptr<Msg> RecvMsg(bool blocked) {
    /*
    shared_ptr<Msg> m = sk_client_->RecvMsg(blocked);
    Dequeue();
    */
    
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

protected:
  // serialize creating new solver within a process
  static boost::mutex       new_solver_mutex_;
  // number of solvers created
  static boost::atomic_int  new_solver_cnt_;

DISABLE_COPY_AND_ASSIGN(WorkerThread);
};


} // end caffe


#endif



