#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
class InternalThread {
 public:
<<<<<<< HEAD
  InternalThread() : thread_() {}
=======
  InternalThread() : thread_(NULL), must_stop_() {}
>>>>>>> origin/BVLC/parallel
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
<<<<<<< HEAD
  void StopInternalThread();
=======
  bool StopInternalThread();
>>>>>>> origin/BVLC/parallel

  bool is_started() const;

  bool must_stop() {
    return must_stop_;
  }

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}

<<<<<<< HEAD
  /* Should be tested when running loops to exit when requested. */
  bool must_stop();

 private:
  void entry(int device, Caffe::Brew mode, int rand_seed, int solver_count,
      bool root_solver);

  shared_ptr<boost::thread> thread_;
=======
  caffe::Thread* thread_;
  bool must_stop_;
>>>>>>> origin/BVLC/parallel
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
