#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#endif

#include <boost/std::shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <math.h>
#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#include "caffe/definitions.hpp"

#ifdef CMAKE_WINDOWS_BUILD
  #include "caffe/export.hpp"
#endif

#include "caffe/util/fp16.hpp"

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

// Instantiate a pointer class
#define INSTANTIATE_POINTER_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<int8_t>; \
  template class classname<uint8_t>; \
  template class classname<int16_t>; \
  template class classname<uint16_t>; \
  template class classname<int32_t>; \
  template class classname<uint32_t>; \
  template class classname<int64_t>; \
  template class classname<uint64_t>; \
  template class classname<half_float::half>; \
  template class classname<float>; \
  template class classname<double>; \
  template class classname<void>;

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS_1T(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>;

#define INSTANTIATE_CLASS_2T(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float, float>; \
  template class classname<double, double>;

#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float, float, float, float>::Forward_gpu( \
      const std::vector<Blob<float, float>*>& bottom, \
      const std::vector<Blob<float, float>*>& top); \
  template void classname<double, double, double, double>::Forward_gpu( \
      const std::vector<Blob<double, double>*>& bottom, \
      const std::vector<Blob<double, double>*>& top);

#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
  template void classname<float>::Backward_gpu( \
      const std::vector<Blob<float>*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob<float>*>& bottom); \
  template void classname<double>::Backward_gpu( \
      const std::vector<Blob<double>*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob<double>*>& bottom)

#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
  INSTANTIATE_LAYER_GPU_FORWARD(classname); \
  INSTANTIATE_LAYER_GPU_BACKWARD(classname)

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

#ifdef CPU_ONLY  // CPU-only Caffe.
// Stub out GPU calls as unavailable.
#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \

#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
#endif

// See PR #1236
namespace cv {class Mat;}

namespace caffe {

class device;

// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::isnan;
using std::isinf;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;
using std::shared_ptr;

// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
void GlobalInit(int* pargc, char*** pargv);

// A singleton class to hold common caffe stuff, such as the handler that
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
    std::shared_ptr<Generator> generator_;
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
  static void set_random_seed(const size_t seed, device* device_context);
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  // Teardown the device
  static void TeardownDevice(const int device_id);
  // Switch the current device
  static void SelectDevice(device* device_context);
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
  static device *GetDefaultDevice();
  static device *GetCPUDevice();

  // Prints info about all devices
  static int EnumerateDevices(bool silent = false);
  // Prepares contexts for devices to use
  static void SetDevices(std::vector<int> device_ids);
  // Finish executing gpu kernels on the specified-device.
  static void Synchronize(int device_id);

  // Get a device context
  static device *GetDevice(int id, bool listId);

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
  std::shared_ptr<RNG> random_generator_;
  Brew mode_;
  // The shared ptrs are being referenced on every thread,
  // while the default device will be handled thread local
  std::shared_ptr<device> cpu_device_;
  device* default_device_;
  static vector<std::shared_ptr< device> > devices_;

  // Parallel training
  int solver_count_;
  int solver_rank_;
  bool multiprocess_;
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
