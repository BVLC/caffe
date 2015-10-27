
#ifndef MULTI_NODE_WORKER_THREAD_H_
#define MULTI_NODE_WORKER_THREAD_H_


#include "caffe/multi_node/msg.hpp"
#include "caffe/multi_node/sk_server.hpp"
#include "caffe/multi_node/node_env.hpp"

#include <boost/thread.hpp>

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
    return sk_client_->RecvMsg(blocked);
  }

  virtual void InternalThreadEntry() {
    sk_client_.reset(new SkSock(ZMQ_PAIR));
    sk_client_->Connect(addr_);

    Run();
  }
  
  /// should be reimplented in the working threads
  virtual void Run() = 0;

protected:
  Solver<Dtype> *NewSolver() {
    boost::mutex::scoped_lock lock(new_solver_mutex_);
    Solver<Dtype> *root_solver = (Solver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
    const vector<Blob<Dtype>*>& root_params = root_solver->net()->learnable_params();

    Solver<Dtype> *new_solver = CreateSolver();
    if (new_solver == NULL) {
      return new_solver;
    }

    const vector<Blob<Dtype>*>& new_params = new_solver->net()->learnable_params();
    
    /// share parameters with root solver
    for (int i = 0; i < new_params.size(); i++) {
      CHECK_EQ(new_params[i]->count(), root_params[i]->count());
      new_params[i]->ShareData(*root_params[i]);
    }

    return new_solver;
  }

  virtual Solver<Dtype> *CreateSolver() = 0;
  
  ///move these functions to caffe::Net?

  /// net get input blob data from message
  static void GetInputData(shared_ptr<Net<Dtype> > net, shared_ptr<Msg> m) {
    for (int i = 0; i < net->num_inputs(); i++) {
      int blob_index = net->input_blob_indices()[i];
      const string& blob_name = net->blob_names()[blob_index];
      Blob<Dtype>* pblob = net->input_blobs()[i];

      m->CopyBlob(blob_name, pblob->mutable_cpu_data(), pblob->count() * sizeof(Dtype));
    }
  }

  /// net get output blob diffs from message
  static void GetOutputDiff(shared_ptr<Net<Dtype> > net, shared_ptr<Msg> m) {
    for (int i = 0; i < net->num_outputs(); i++) {
      int blob_index = net->output_blob_indices()[i];
      const string& blob_name = net->blob_names()[blob_index];
      Blob<Dtype>* pblob = net->output_blobs()[i];

      m->CopyBlob(blob_name, pblob->mutable_cpu_diff(), pblob->count() * sizeof(Dtype));
    }
  }
  
  /// net copy input blob diffs to message
  static void CopyInputDiff(shared_ptr<Net<Dtype> > net, shared_ptr<Msg> m) {
    for (int i = 0; i < net->num_inputs(); i++) {
      int blob_index = net->input_blob_indices()[i];
      const string& blob_name = net->blob_names()[blob_index];
      Blob<Dtype>* pblob = net->input_blobs()[i];

      m->AddNewBlob(blob_name, pblob->cpu_diff(), pblob->count() * sizeof(Dtype));
    }
  }
  
  /// net copy output blob data to a message
  static void CopyOutputData(shared_ptr<Net<Dtype> > net, shared_ptr<Msg> m) {
    for (int i = 0; i < net->num_outputs(); i++) {
      int blob_index = net->output_blob_indices()[i];
      const string& blob_name = net->blob_names()[blob_index];
      Blob<Dtype>* pblob = net->output_blobs()[i];

      m->AddNewBlob(blob_name, pblob->cpu_data(), pblob->count() * sizeof(Dtype));
    }
  }
  

protected:
  int worker_id_;
  /// in-process communication addr with routing thread
  string addr_;
  shared_ptr<SkSock> sk_client_;

protected:
  //serialize creating new solver within a process
  static boost::mutex  new_solver_mutex_;

DISABLE_COPY_AND_ASSIGN(WorkerThread);
};


} // end caffe


#endif



