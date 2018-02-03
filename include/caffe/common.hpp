#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#endif

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/enum.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "caffe/definitions.hpp"
#include "caffe/macros.hpp"

#ifdef CMAKE_WINDOWS_BUILD
  #include "caffe/export.hpp"
#endif

#ifdef USE_CUDA
#include "caffe/backend/cuda/caffe_cuda.hpp"
#endif

#ifdef USE_OPENCL
#include "caffe/backend/opencl/caffe_opencl.hpp"
#endif

#ifdef USE_HIP
#include "caffe/backend/hip/caffe_hip.hpp"
#endif

// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFLAGS_H_ to detect if it is version
// 2.1. If yes, we will add a temporary solution to redirect the namespace.
// TODO(Yangqing): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)


// a simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// See PR #1236
namespace cv {class Mat;}

namespace caffe {

class Device;

// a global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
void GlobalInit(int* pargc, char*** pargv);

// a singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Caffe {
 public:
  Caffe();
  Caffe(const Caffe &obj);
  ~Caffe();

  // Thread local context for Caffe. Moved to common.cpp instead of
  // including boost/thread.hpp to avoid a boost/NVCC issues (#1009, #1010)
  // on OSX. Also fails on Linux with CUDA 7.0.18.
  static Caffe& Get();

  enum Brew { CPU, GPU };

  // This random number generator facade hides boost and CUDA rng
  // implementation from one another (for cross-platform compatibility).
  class RNG {
   public:
    RNG();
    explicit RNG(size_t);
    explicit RNG(const RNG&);
    RNG& operator=(const RNG&);
    void* generator();
   private:
    class Generator;
    shared_ptr<Generator> generator_;
  };

  // Getters for boost rng, curand, and cublas handles
  inline static RNG& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    return *(Get().random_generator_);
  }
#ifndef CPU_ONLY
#ifdef USE_CUDA
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  inline static curandGenerator_t curand_generator() {
    return Get().curand_generator_;
  }
  inline static curandGenerator_t curand_generator64() {
    return Get().curand_generator64_;
  }
#endif  // USE_CUDA
#if defined(USE_OPENCL) && defined(USE_FFT)
  inline static ClFFTState& cl_fft_state() { return Get().cl_fft_state_; }
#endif  // USE_OPENCL
#endif  // !CPU_ONLY

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the random seed of both boost and curand
  static void set_random_seed(const size_t seed, Device* device_context);
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  // Teardown the device
  static void TeardownDevice(const int device_id);
  // Switch the current device
  static void SelectDevice(Device* device_context);
  static void SelectDevice(int id, bool listId);

  // Prints the current GPU status.
  static void DeviceQuery();
  // Check if specified device is available
  static bool CheckDevice(const int device_id);
  // Search from start_id to the highest possible device ordinal,
  // return the ordinal of the first available device.
  static int FindDevice(const int start_id = 0);
  // Parallel training
  inline static int solver_count() { return Get().solver_count_; }
  inline static void set_solver_count(int val) { Get().solver_count_ = val; }
  inline static int solver_rank() { return Get().solver_rank_; }
  inline static void set_solver_rank(int val) { Get().solver_rank_ = val; }
  inline static bool multiprocess() { return Get().multiprocess_; }
  inline static void set_multiprocess(bool val) { Get().multiprocess_ = val; }
  inline static bool root_solver() { return Get().solver_rank_ == 0; }

  // Get the default device
  static Device *GetDefaultDevice();
  static Device *GetCPUDevice();

  // Prints info about all devices
  static int EnumerateDevices(bool silent = false);
  // Prepares contexts for devices to use
  static void SetDevices(vector<int> device_ids);

#ifdef USE_OPENCL
  static const cl_context& GetOpenCLContext(const int id, bool list_id);
  static const cl_command_queue& GetOpenCLQueue(const int id, bool list_id);
#endif  // USE_OPENCL

  // Finish executing gpu kernels on the specified-device.
  static void Synchronize(int device_id);

  // Get a device context
  static Device *GetDevice(int id, bool listId);

 protected:
#ifndef CPU_ONLY
#ifdef USE_CUDA
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
  curandGenerator_t curand_generator64_;
#endif  // USE_CUDA
#if defined(USE_OPENCL) && defined(USE_FFT)
  ClFFTState cl_fft_state_;
#endif
#endif  // !CPU_ONLY
  shared_ptr<RNG> random_generator_;
  Brew mode_;
  // The shared ptrs are being referenced on every thread,
  // while the default device will be handled thread local
  shared_ptr<Device> cpu_device_;
  Device* default_device_;
  static vector<shared_ptr<Device> > devices_;

  // Parallel training
  int solver_count_;
  int solver_rank_;
  bool multiprocess_;
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
