#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <memory>
//#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream> // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility> // pair
#include <vector>

#include "caffe/util/device_alternate.hpp"

namespace deepir {
namespace allocator {
class buddy_pool;
}
} // namespace deepir

// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

/*
// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFLAGS_H_ to detect if it is version
// 2.1. If yes, we will add a temporary solution to redirect the namespace.
// TODO(Yangqing): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_
*/

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname)                                     \
private:                                                                       \
  classname(const classname &);                                                \
  classname &operator=(const classname &)

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname)                                           \
  char gInstantiationGuard##classname;                                         \
  template class classname<float>;                                             \
  template class classname<double>

#define INSTANTIATE_LAYER_GPU_FORWARD(classname)                               \
  template void classname<float>::Forward_gpu(                                 \
      const std::vector<Blob<float> *> &bottom,                                \
      const std::vector<Blob<float> *> &top);                                  \
  template void classname<double>::Forward_gpu(                                \
      const std::vector<Blob<double> *> &bottom,                               \
      const std::vector<Blob<double> *> &top);

#define INSTANTIATE_LAYER_GPU_FORWARD_CONST(classname)                         \
  template void classname<float>::Forward_const_gpu(                           \
      const std::vector<Blob<float> *> &bottom,                                \
      const std::vector<Blob<float> *> &top) const;                            \
  template void classname<double>::Forward_const_gpu(                          \
      const std::vector<Blob<double> *> &bottom,                               \
      const std::vector<Blob<double> *> &top) const;

#define INSTANTIATE_LAYER_GPU_FUNCS(classname)                                 \
  INSTANTIATE_LAYER_GPU_FORWARD(classname);

#define INSTANTIATE_LAYER_GPU_FUNCS_CONST(classname)                           \
  INSTANTIATE_LAYER_GPU_FORWARD(classname);                                    \
  INSTANTIATE_LAYER_GPU_FORWARD_CONST(classname);

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// See PR #1236
namespace cv {
class Mat;
}

namespace caffe {

using std::shared_ptr;

// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::isinf;
using std::isnan;
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
void GlobalInit(int *pargc, char ***pargv);

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Caffe {
public:
  ~Caffe();

  static Caffe &Get();

  enum Brew { CPU, GPU };

#ifndef CPU_ONLY
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  inline static const std::shared_ptr<deepir::allocator::buddy_pool> &
  host_pool() {
    return Get().host_pool_;
  }
  inline static const std::shared_ptr<deepir::allocator::buddy_pool> &
  device_pool() {
    return Get().device_pool_;
  }

#endif

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  static void set_mode(Brew mode);
  static int GetDevice() { return Get().device_id_; }
private:
#ifndef CPU_ONLY
  cublasHandle_t cublas_handle_;
  std::shared_ptr<deepir::allocator::buddy_pool> host_pool_;
  std::shared_ptr<deepir::allocator::buddy_pool> device_pool_;
#endif

  Brew mode_;
  int device_id_;

private:
  // The private constructor to avoid duplicate instantiation.
  Caffe();

  DISABLE_COPY_AND_ASSIGN(Caffe);
};

} // namespace caffe

#endif // CAFFE_COMMON_HPP_
