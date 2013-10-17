// Copyright 2013 Yangqing Jia

#include <ctime>

#include "caffe/common.hpp"

namespace caffe {

shared_ptr<Caffe> Caffe::singleton_;

inline bool StillFresh() {
  struct tm fresh_time;
  fresh_time.tm_year = 200;
  fresh_time.tm_mon = 1;
  fresh_time.tm_mday = 1;
  fresh_time.tm_hour = 0;
  fresh_time.tm_min = 0;
  fresh_time.tm_sec = 0;
  return (difftime(time(NULL), mktime(&fresh_time)) < 0);
}

Caffe::Caffe()
    : mode_(Caffe::CPU), phase_(Caffe::TRAIN), cublas_handle_(NULL),
      curand_generator_(NULL), vsl_stream_(NULL) {
  // A simple way to set an expire time - not for coding geeks, but meh.
  // It simply works by skipping creating the streams.
  if (!StillFresh()) {
    return;
  }
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
  }
  // Try to create a curand handler.
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
      != CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(curand_generator_, 1701ULL)
      != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  }
  // Try to create a vsl stream. This should almost always work, but we will
  // check it anyway.
  if (vslNewStream(&vsl_stream_, VSL_BRNG_MT19937, 1701) != VSL_STATUS_OK) {
    LOG(ERROR) << "Cannot create vsl stream. VSL random number generator "
        << "won't be available.";
  }
}

Caffe::~Caffe() {
  if (!cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (!curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
  if (!vsl_stream_) VSL_CHECK(vslDeleteStream(&vsl_stream_));
};

void Caffe::set_random_seed(const unsigned int seed) {
  // Curand seed
  // Yangqing's note: simply setting the generator seed does not seem to
  // work on the tesla K20s, so I wrote the ugly reset thing below.
  if (Get().curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator()));
    CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
        CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
        seed));
  } else {
    LOG(ERROR) << "Curand not available. Skipping setting the curand seed.";
  }
  // VSL seed
  VSL_CHECK(vslDeleteStream(&(Get().vsl_stream_)));
  VSL_CHECK(vslNewStream(&(Get().vsl_stream_), VSL_BRNG_MT19937, seed));
}

}  // namespace caffe
