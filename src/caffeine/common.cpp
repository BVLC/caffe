#include "caffeine/common.hpp"

namespace caffeine {

shared_ptr<Caffeine> Caffeine::singleton_;

Caffeine::Caffeine()
    : mode_(Caffeine::CPU) {
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  VSL_CHECK(vslNewStream(&vsl_stream_, VSL_BRNG_MT19937, 1701));
}

Caffeine::~Caffeine() {
  if (!cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (!vsl_stream_) VSL_CHECK(vslDeleteStream(&vsl_stream_));
};

Caffeine& Caffeine::Get() {
  if (!singleton_) {
    singleton_.reset(new Caffeine());
  }
  return *singleton_;
};

VSLStreamStatePtr Caffeine::vsl_stream() {
  return Get().vsl_stream_;
}

cublasHandle_t Caffeine::cublas_handle() {
  return Get().cublas_handle_;
};

Caffeine::Brew Caffeine::mode() {
  return Get().mode_;
}


void Caffeine::set_mode(Caffeine::Brew mode) {
  Get().mode_ = mode;
}

}  // namespace caffeine

