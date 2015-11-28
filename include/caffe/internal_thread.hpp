#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge

namespace caffe {

/**
<<<<<<< HEAD
<<<<<<< HEAD
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
<<<<<<< HEAD
 */
class InternalThread {
 public:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  InternalThread() : thread_() {}
=======
  InternalThread() : thread_(NULL), must_stop_() {}
>>>>>>> origin/BVLC/parallel
=======
  InternalThread() : thread_(NULL), must_stop_() {}
>>>>>>> origin/BVLC/parallel
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
<<<<<<< HEAD
<<<<<<< HEAD
  void StopInternalThread();
=======
  bool StopInternalThread();
>>>>>>> origin/BVLC/parallel

  bool is_started() const;
=======
  bool StopInternalThread();
>>>>>>> origin/BVLC/parallel

  bool must_stop() {
    return must_stop_;
  }
=======
  bool StopInternalThread();
>>>>>>> origin/BVLC/parallel

  bool must_stop() {
    return must_stop_;
  }
=======
<<<<<<< HEAD
<<<<<<< HEAD
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  void StopInternalThread();

  bool is_started() const;
=======
 * A minimal wrapper for boost::thread to force host compilation for boost
 * Defined in caffe/util/thread.hpp
 */
class Thread {
 public:
  template<typename Callable, class A1>
  Thread(Callable func, A1 a1);
  void join();
  bool joinable();
 private:
  void* thread_;
};

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virutal function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_(NULL), must_stop_() {}
  virtual ~InternalThread();

  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool StopInternalThread();

  bool is_started() const { return thread_ != NULL && thread_->joinable(); }
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  void StopInternalThread();

  bool is_started() const;
=======
 * A minimal wrapper for boost::thread to force host compilation for boost
 * Defined in caffe/util/thread.hpp
 */
class Thread {
 public:
  template<typename Callable, class A1>
  Thread(Callable func, A1 a1);
  void join();
  bool joinable();
 private:
  void* thread_;
};

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virutal function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_(NULL), must_stop_() {}
  virtual ~InternalThread();

=======
 */
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  void StopInternalThread();

  bool is_started() const;
=======
 * A minimal wrapper for boost::thread to force host compilation for boost
 * Defined in caffe/util/thread.hpp
 */
class Thread {
 public:
  template<typename Callable, class A1>
  Thread(Callable func, A1 a1);
  void join();
  bool joinable();
 private:
  void* thread_;
};

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virutal function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_(NULL), must_stop_() {}
  virtual ~InternalThread();

>>>>>>> pod/caffe-merge
  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool StopInternalThread();

  bool is_started() const { return thread_ != NULL && thread_->joinable(); }
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge

  bool must_stop() {
    return must_stop_;
  }
>>>>>>> origin/BVLC/parallel
=======
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  void StopInternalThread();

  bool is_started() const;
>>>>>>> caffe

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> caffe

<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe

>>>>>>> pod/caffe-merge
  /* Should be tested when running loops to exit when requested. */
  bool must_stop();

 private:
  void entry(int device, Caffe::Brew mode, int rand_seed, int solver_count,
      bool root_solver);
<<<<<<< HEAD

  shared_ptr<boost::thread> thread_;
=======
<<<<<<< HEAD
<<<<<<< HEAD
  caffe::Thread* thread_;
  bool must_stop_;
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD

  shared_ptr<boost::thread> thread_;
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge

  caffe::Thread* thread_;
  bool must_stop_;
>>>>>>> origin/BVLC/parallel
=======

  shared_ptr<boost::thread> thread_;
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
