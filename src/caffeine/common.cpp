#include "caffeine/common.hpp"

namespace caffeine {

shared_ptr<Caffeine> Caffeine::singleton_;

Caffeine::Caffeine()
    : mode_(Caffeine::CPU) {
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
}

Caffeine::~Caffeine() {
  if (!cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
};

Caffeine& Caffeine::Get() {
  if (!singleton_) {
    singleton_.reset(new Caffeine());
  }
  return *singleton_;
};

cublasHandle_t Caffeine::cublas_handle() {
  return Get().cublas_handle_;
};

Caffeine::Brew Caffeine::mode() {
  return Get().mode_;
}

Caffeine::Brew Caffeine::set_mode(Caffeine::Brew mode) {
  Get().mode_ = mode;
}

}  // namespace caffeine
