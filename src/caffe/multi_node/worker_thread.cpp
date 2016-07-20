

#include "caffe/multi_node/worker_thread.hpp"
#include "caffe/multi_node/node_env.hpp"

namespace caffe {
template <typename Dtype>
boost::mutex  WorkerThread<Dtype>::new_solver_mutex_;

template <typename Dtype>
boost::atomic_int WorkerThread<Dtype>::new_solver_cnt_(0);

INSTANTIATE_CLASS(WorkerThread);

} // end caffe


