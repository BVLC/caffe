#ifndef CAFFEINE_COMMON_HPP_
#define CAFFEINE_COMMON_HPP_

#include <iostream>

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
 private:
  Caffeine() {
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  };
  static shared_ptr<Caffeine> singleton_;
  cublasHandle_t cublas_handle_;
 public:
  ~Caffeine() {
    if (!cublas_handle_) {
      CUBLAS_CHECK(cublasDestroy(cublas_handle_));
    }
  }
  static Caffeine& Get() {
    if (!singleton_) {
      singleton_.reset(new Caffeine());
    }
    return *singleton_;
  }
  
  static cublasHandle_t cublas_handle() {
    return Get().cublas_handle_;
  }
};

}


#endif  // CAFFEINE_COMMON_HPP_
