#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#include "caffe/util/device_alternate.hpp"

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

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::Forward_gpu( \
      const std::vector<Blob<float>*>& bottom, \
      const std::vector<Blob<float>*>& top); \
  template void classname<double>::Forward_gpu( \
      const std::vector<Blob<double>*>& bottom, \
      const std::vector<Blob<double>*>& top);

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

// See PR #1236
namespace cv { class Mat; }

namespace caffe {

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;

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

// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
void GlobalInit(int* pargc, char*** pargv);

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Caffe {
 public:
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
    explicit RNG(unsigned int seed);
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
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  inline static curandGenerator_t curand_generator() {
    return Get().curand_generator_;
  }
#endif

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the random seed of both boost and curand
  static void set_random_seed(const unsigned int seed);
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  // Prints the current GPU status.
  static void DeviceQuery();
  // Parallel training info
  inline static int solver_count() { return Get().solver_count_; }
  inline static void set_solver_count(int val) { Get().solver_count_ = val; }
  inline static bool root_solver() { return Get().root_solver_; }
  inline static void set_root_solver(bool val) { Get().root_solver_ = val; }

 protected:
#ifndef CPU_ONLY
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
#endif
  shared_ptr<RNG> random_generator_;

  Brew mode_;
  int solver_count_;
  bool root_solver_;

 private:
  // The private constructor to avoid duplicate instantiation.
  Caffe();

  DISABLE_COPY_AND_ASSIGN(Caffe);
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
