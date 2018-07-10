#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

InternalThread::~InternalThread() {
  StopInternalThread();
}

bool InternalThread::is_started() const {
  return thread_ && thread_->joinable();
}

bool InternalThread::must_stop() {
  return thread_ && thread_->interruption_requested();
}

void InternalThread::StartInternalThread(Device* dev) {
  CHECK(!is_started()) << "Threads should persist and not be restarted.";

  thread_device_ = dev;

  Caffe::Brew mode = Caffe::mode();
  int rand_seed = caffe_rng_rand();
  int solver_count = Caffe::solver_count();
  int solver_rank = Caffe::solver_rank();
  bool multiprocess = Caffe::multiprocess();

  try {
    thread_.reset(new boost::thread(&InternalThread::entry, this, dev, mode,
          rand_seed, solver_count, solver_rank, multiprocess));
  } catch (std::exception& e) {
    LOG(FATAL)<< "Thread exception: " << e.what();
  }
}

void InternalThread::entry(Device* dev, Caffe::Brew mode, int_tp rand_seed,
    int_tp solver_count, int_tp solver_rank, bool multiprocess) {
  Caffe::SelectDevice(dev);
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed, thread_device_);
  Caffe::set_solver_count(solver_count);
  Caffe::set_solver_rank(solver_rank);
  Caffe::set_multiprocess(multiprocess);

  InternalThreadEntry();
}

void InternalThread::StopInternalThread() {
  if (is_started()) {
    thread_->interrupt();
    try {
      thread_->join();
    } catch (boost::thread_interrupted&) {
    } catch (std::exception& e) {
      LOG(FATAL) << "Thread exception: " << e.what();
    }
  }
}

}  // namespace caffe
