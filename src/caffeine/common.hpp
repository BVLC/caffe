#ifndef CAFFEINE_COMMON_HPP_
#define CAFFEINE_COMMON_HPP_

#include <boost/shared_ptr.hpp>
#include <cublas_v2.h>
#include <glog/logging.h>

#include "driver_types.h"

namespace caffeine {

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;

#define CUDA_CHECK(condition) CHECK((condition) == cudaSuccess)
#define CUBLAS_CHECK(condition) CHECK((condition) == CUBLAS_STATUS_SUCCESS)

// A singleton class to hold common caffeine stuff, such as the handler that
// caffeine is going to use for cublas.
class Caffeine {
 public:
  ~Caffeine();
  static Caffeine& Get();
  enum Brew { CPU, GPU };

  // The getters for the variables. 
  static cublasHandle_t cublas_handle();
  static Brew mode();
  // The setters for the variables
  static Brew set_mode(Brew mode);
 private:
  Caffeine();
  static shared_ptr<Caffeine> singleton_;
  cublasHandle_t cublas_handle_;
  Brew mode_;
};

}

#endif  // CAFFEINE_COMMON_HPP_
