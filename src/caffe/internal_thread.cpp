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

void InternalThread::StartInternalThread(device* device_context) {
  CHECK(!is_started()) << "Threads should persist and not be restarted.";

  thread_device_ = device_context;

  Caffe::Brew mode = Caffe::mode();
  int_tp rand_seed = caffe_rng_rand();
  int_tp solver_count = Caffe::solver_count();
  bool root_solver = Caffe::root_solver();

  try {
    thread_.reset(
        new boost::thread(&InternalThread::entry, this, thread_device_,
                          mode, rand_seed, solver_count, root_solver));
  } catch (std::exception& e) {
    LOG(FATAL)<< "Thread exception: " << e.what();
  }
}

void InternalThread::entry(device* device_context, Caffe::Brew mode,
                           int_tp rand_seed,
                           int_tp solver_count, bool root_solver) {
  Caffe::SelectDevice(device_context);
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed, thread_device_);
  Caffe::set_solver_count(solver_count);
  Caffe::set_root_solver(root_solver);

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
