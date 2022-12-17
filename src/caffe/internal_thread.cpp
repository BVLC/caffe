#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
// 析构函数，调用停止内部线程函数
InternalThread::~InternalThread() {
  StopInternalThread();
}

// 测试线程是否起来
bool InternalThread::is_started() const {
  return thread_ && thread_->joinable();
}

bool InternalThread::must_stop() {
  return thread_ && thread_->interruption_requested();
}

void InternalThread::StartInternalThread() {
  CHECK(!is_started()) << "Threads should persist and not be restarted.";

  int device = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device));
#endif
  Caffe::Brew mode = Caffe::mode();
  int rand_seed = caffe_rng_rand();
  int solver_count = Caffe::solver_count();
  int solver_rank = Caffe::solver_rank();
  bool multiprocess = Caffe::multiprocess();

  try {
    // 然后重新实例化一个thread对象给thread_指针，该线程的执行的是entry函数
    // boost::thread输入依次为函数名和传入的参数
    thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
          rand_seed, solver_count, solver_rank, multiprocess));
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

//  
void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
    int solver_count, int solver_rank, bool multiprocess) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device));
#endif
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);
  Caffe::set_solver_rank(solver_rank);
  Caffe::set_multiprocess(multiprocess);

  InternalThreadEntry();
}

void InternalThread::StopInternalThread() {
  if (is_started()) {
    thread_->interrupt(); // 打断线程
    try {
      thread_->join(); //等待线程结束
    } catch (boost::thread_interrupted&) { //如果被打断，啥也不干，因为是自己要打断的^
    } catch (std::exception& e) {
      LOG(FATAL) << "Thread exception: " << e.what();
    }
  }
}

}  // namespace caffe
