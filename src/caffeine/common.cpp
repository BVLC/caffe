#include "caffeine/common.hpp"

namespace caffeine {

shared_ptr<Caffeine> Caffeine::singleton_;

Caffeine::Caffeine()
    : mode_(Caffeine::CPU), phase_(Caffeine::TRAIN) {
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CURAND_CHECK(curandCreateGenerator(&curand_generator_,
      CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_,
      1701ULL));
  VSL_CHECK(vslNewStream(&vsl_stream_, VSL_BRNG_MT19937, 1701));
}

Caffeine::~Caffeine() {
  if (!cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (!curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
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

curandGenerator_t Caffeine::curand_generator() {
  return Get().curand_generator_;
};

Caffeine::Brew Caffeine::mode() {
  return Get().mode_;
}

void Caffeine::set_mode(Caffeine::Brew mode) {
  Get().mode_ = mode;
}

Caffeine::Phase Caffeine::phase() {
  return Get().phase_;
}

void Caffeine::set_phase(Caffeine::Phase phase) {
  Get().phase_ = phase;
}

void Caffeine::set_random_seed(const unsigned int seed) {
  // Curand seed
  // Yangqing's note: simply setting the generator seed does not seem to
  // work on the tesla K20s, so I wrote the ugly reset thing below. It is not
  // tested yet and I'll wait til Jeff finishes training.
  CURAND_CHECK(curandDestroyGenerator(curand_generator()));
  CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
      CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
      (unsigned long long)seed));
  // VSL seed
  VSL_CHECK(vslDeleteStream(&(Get().vsl_stream_)));
  VSL_CHECK(vslNewStream(&(Get().vsl_stream_), VSL_BRNG_MT19937, seed));
}

}  // namespace caffeine

