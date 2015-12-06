#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
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
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge

namespace caffe {

/**
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
<<<<<<< HEAD
<<<<<<< HEAD
 */
class InternalThread {
 public:
<<<<<<< HEAD
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
<<<<<<< HEAD
  void StopInternalThread();
=======
  bool StopInternalThread();
>>>>>>> origin/BVLC/parallel
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
=======
>>>>>>> pod/caffe-merge
=======
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
>>>>>>> pod/caffe-merge
 */
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

<<<<<<< HEAD
=======
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

>>>>>>> pod/device/blob.hpp
  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  void StartInternalThread();
<<<<<<< HEAD

  /** Will not return until the internal thread has exited. */
  void StopInternalThread();
=======

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
>>>>>>> pod/device/blob.hpp

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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool StopInternalThread();

  bool is_started() const { return thread_ != NULL && thread_->joinable(); }
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
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
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe

>>>>>>> pod-caffe-pod.hpp-merge
  /* Should be tested when running loops to exit when requested. */
  bool must_stop();
>>>>>>> pod/device/blob.hpp

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
<<<<<<< HEAD
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
  void entry(int device, Caffe::Brew mode, int rand_seed, int solver_count,
      bool root_solver);
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD

=======
=======
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
>>>>>>> pod-caffe-pod.hpp-merge
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

<<<<<<< HEAD
>>>>>>> pod/caffe-merge
  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool StopInternalThread();

  bool is_started() const { return thread_ != NULL && thread_->joinable(); }
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/caffe-merge

  bool must_stop() {
    return must_stop_;
  }
>>>>>>> origin/BVLC/parallel
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
>>>>>>> caffe
=======

  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool StopInternalThread();

  bool is_started() const { return thread_ != NULL && thread_->joinable(); }
>>>>>>> pod/caffe-merge
=======

  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool StopInternalThread();

  bool is_started() const { return thread_ != NULL && thread_->joinable(); }
>>>>>>> pod-caffe-pod.hpp-merge

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
<<<<<<< HEAD
=======
  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool StopInternalThread();

  bool is_started() const { return thread_ != NULL && thread_->joinable(); }
>>>>>>> pod-caffe-pod.hpp-merge

  bool must_stop() {
    return must_stop_;
  }
<<<<<<< HEAD
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
=======

  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();
>>>>>>> pod/caffe-merge

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
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
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

<<<<<<< HEAD
<<<<<<< HEAD
=======
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  void StopInternalThread();

  bool is_started() const;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> caffe

  bool must_stop() {
    return must_stop_;
  }
=======
>>>>>>> device-abstraction
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge

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
=======
>>>>>>> pod-caffe-pod.hpp-merge

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp

<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
>>>>>>> caffe

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe

>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> caffe

>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======

>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe

>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe

>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe

>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe

>>>>>>> pod-caffe-pod.hpp-merge
  /* Should be tested when running loops to exit when requested. */
  bool must_stop();

 private:
  void entry(int device, Caffe::Brew mode, int rand_seed, int solver_count,
      bool root_solver);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp

  shared_ptr<boost::thread> thread_;
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  caffe::Thread* thread_;
  bool must_stop_;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
=======
>>>>>>> pod/common.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD

  shared_ptr<boost::thread> thread_;
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======

  shared_ptr<boost::thread> thread_;
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======

  shared_ptr<boost::thread> thread_;
=======
>>>>>>> pod/caffe-merge
=======

  shared_ptr<boost::thread> thread_;
=======
>>>>>>> pod/device/blob.hpp
=======

  shared_ptr<boost::thread> thread_;
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD

  shared_ptr<boost::thread> thread_;
=======
>>>>>>> pod/device/blob.hpp

  caffe::Thread* thread_;
  bool must_stop_;
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

  shared_ptr<boost::thread> thread_;
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======

  shared_ptr<boost::thread> thread_;
>>>>>>> device-abstraction
=======
=======
>>>>>>> pod/device/blob.hpp

  shared_ptr<boost::thread> thread_;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======

  shared_ptr<boost::thread> thread_;
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp

  shared_ptr<boost::thread> thread_;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
