#ifndef CPU_ONLY
#include <map>
#include "caffe/common.hpp"
#if defined(USE_GREENTEA) && defined(USE_FFT)
#include "caffe/util/cl_fft_state.hpp"

namespace caffe {

ClFFTState::ClFFTState()
  : initialized_(false) {
}

void ClFFTState::setup() {
  if (!initialized_) {
    clfftSetupData fftSetup;
    CLFFT_CHECK(clfftInitSetupData(&fftSetup));
    CLFFT_CHECK(clfftSetup(&fftSetup));
    LOG(INFO) << "Setup clFFT";
    initialized_ = true;
  }
}

void ClFFTState::teardown() {
  if (!initialized_) {
    LOG(INFO) << "clfft does not setup.";
    return;
  }
  std::map<KeyType, clfftPlanHandle>::iterator it;
  for (it = forward_fft_inplace_many_handle_map_.begin();
      it != forward_fft_inplace_many_handle_map_.end(); ++it) {
    CLFFT_CHECK(clfftDestroyPlan(&(it->second)));
  }
  forward_fft_inplace_many_handle_map_.clear();
  for (it = forward_fft_many_handle_map_.begin();
       it != forward_fft_many_handle_map_.end();
      ++it) {
    CLFFT_CHECK(clfftDestroyPlan(&(it->second)));
  }
  forward_fft_many_handle_map_.clear();
  for (it = backward_fft_many_handle_map_.begin();
       it != backward_fft_many_handle_map_.end(); ++it) {
    CLFFT_CHECK(clfftDestroyPlan(&(it->second)));
  }
  backward_fft_many_handle_map_.clear();
  for (it = forward_ifft_many_handle_map_.begin();
       it != forward_ifft_many_handle_map_.end();
      ++it) {
    CLFFT_CHECK(clfftDestroyPlan(&(it->second)));
  }
  forward_ifft_many_handle_map_.clear();
  for (it = backward_ifft_many_handle_map_.begin();
       it != backward_ifft_many_handle_map_.end();
      ++it) {
    CLFFT_CHECK(clfftDestroyPlan(&(it->second)));
  }
  backward_ifft_many_handle_map_.clear();

  CLFFT_CHECK(clfftTeardown());
  LOG(INFO) << "Teardown clFFT";

  initialized_ = false;
}

clfftPlanHandle ClFFTState::getForwardInPlaceFFTManyPlanHandle(
    const int height, const int width, const int batch_size) {
  if (!initialized_) {
    LOG(INFO) << "clfft does not setup.";
    return (clfftPlanHandle)NULL;
  }
  std::map<KeyType, clfftPlanHandle>::iterator it =
      forward_fft_inplace_many_handle_map_.find(KeyType(FFTSize(height, width),
          batch_size));
  if (it != forward_fft_inplace_many_handle_map_.end()) {
    return it->second;
  }
  clfftPlanHandle handle = createInPlaceManyPlanHandle(height, width,
      batch_size, CLFFT_FORWARD);
  if (handle) {
    forward_fft_inplace_many_handle_map_.insert(
        KeyType_HandlePtr(KeyType(FFTSize(height, width), batch_size), handle));
  }
  return handle;
}

clfftPlanHandle ClFFTState::getForwardOutOfPlaceFFTManyPlanHandle(
    const int height, const int width, const int batch_size) {
  if (!initialized_) {
    LOG(INFO) << "clfft does not setup.";
    return (clfftPlanHandle)NULL;
  }
  std::map<KeyType, clfftPlanHandle>::iterator it =
      forward_fft_many_handle_map_.find(KeyType(FFTSize(height, width),
          batch_size));
  if (it != forward_fft_many_handle_map_.end()) {
    return it->second;
  }
  clfftPlanHandle handle = createOutOfPlaceManyPlanHandle(height, width,
      batch_size, CLFFT_FORWARD);
  if (handle) {
    forward_fft_many_handle_map_.insert(
        KeyType_HandlePtr(KeyType(FFTSize(height, width), batch_size), handle));
  }
  return handle;
}

clfftPlanHandle ClFFTState::getBackwardOutOfPlaceFFTManyPlanHandle(
    const int height, const int width, const int batch_size) {
  if (!initialized_) {
    LOG(INFO) << "clfft does not setup.";
    return (clfftPlanHandle)NULL;
  }
  std::map<KeyType, clfftPlanHandle>::iterator it =
      backward_fft_many_handle_map_.find(KeyType(FFTSize(height, width),
          batch_size));
  if (it != backward_fft_many_handle_map_.end()) {
    return it->second;
  }
  clfftPlanHandle handle = createOutOfPlaceManyPlanHandle(height, width,
      batch_size, CLFFT_FORWARD);
  if (handle) {
    backward_fft_many_handle_map_.insert(
        KeyType_HandlePtr(KeyType(FFTSize(height, width), batch_size), handle));
  }
  return handle;
}

clfftPlanHandle ClFFTState::getForwardOutOfPlaceIFFTManyPlanHandle(
    const int height, const int width, const int batch_size) {
  if (!initialized_) {
    LOG(INFO) << "clfft does not setup.";
    return (clfftPlanHandle)NULL;
  }
  std::map<KeyType, clfftPlanHandle>::iterator it =
      forward_ifft_many_handle_map_.find(KeyType(FFTSize(height, width),
          batch_size));
  if (it != forward_ifft_many_handle_map_.end()) {
    return it->second;
  }
  clfftPlanHandle handle = createOutOfPlaceManyPlanHandle(height, width,
      batch_size, CLFFT_BACKWARD);
  if (handle) {
    forward_ifft_many_handle_map_.insert(
        KeyType_HandlePtr(KeyType(FFTSize(height, width), batch_size), handle));
  }
  return handle;
}

clfftPlanHandle ClFFTState::getBackwardOutOfPlaceIFFTManyPlanHandle(
    const int height, const int width, const int batch_size) {
  if (!initialized_) {
    LOG(INFO) << "clfft does not setup.";
    return (clfftPlanHandle)NULL;
  }
  std::map<KeyType, clfftPlanHandle>::iterator it =
      backward_ifft_many_handle_map_.find(KeyType(FFTSize(height, width),
          batch_size));
  if (it != backward_ifft_many_handle_map_.end()) {
    return it->second;
  }
  clfftPlanHandle handle = createOutOfPlaceManyPlanHandle(height, width,
      batch_size, CLFFT_BACKWARD);
  if (handle) {
    backward_ifft_many_handle_map_.insert(
        KeyType_HandlePtr(KeyType(FFTSize(height, width), batch_size), handle));
  }
  return handle;
}

clfftPlanHandle ClFFTState::createOutOfPlaceManyPlanHandle(int height,
    int width, int batch_size, clfftDirection dir) {
  if (!initialized_) {
    LOG(INFO) << "clfft does not setup.";
    return (clfftPlanHandle)NULL;
  }
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();
  viennacl::ocl::command_queue &queue = ctx.get_queue();

  clfftPlanHandle handle;
  float scale = 1.f;
  int idist, odist;
  size_t instrides[2], outstrides[2];
  size_t lengths[2] = { (size_t)width, (size_t)height };
  CLFFT_CHECK(clfftCreateDefaultPlan(&handle, ctx.handle().get(),
              CLFFT_2D, lengths));

  if (CLFFT_FORWARD == dir) {  // FFT plan handle
    idist = height * width;
    odist = height * (width/2 + 1);
    instrides[0] = 1;
    instrides[1] = width;
    outstrides[0] = 1;
    outstrides[1] = (width/2 + 1);
    CLFFT_CHECK(clfftSetLayout(handle, CLFFT_REAL,
        CLFFT_HERMITIAN_INTERLEAVED));
  } else if (CLFFT_BACKWARD == dir) {  // Inverse FFT plan handle
    scale = 1.f / static_cast<float>(height * width);
    idist = height * (width/2 + 1);
    odist = height * width;
    instrides[0] = 1;
    instrides[1] = (width/2 + 1);
    outstrides[0] = 1;
    outstrides[1] = width;
    CLFFT_CHECK(clfftSetLayout(handle, CLFFT_HERMITIAN_INTERLEAVED,
        CLFFT_REAL));
  } else {
    CLFFT_CHECK(clfftDestroyPlan(&handle));
    LOG(ERROR) << "Not implemented";
    return (clfftPlanHandle)NULL;
  }

  CLFFT_CHECK(clfftSetResultLocation(handle, CLFFT_OUTOFPLACE));
  CLFFT_CHECK(clfftSetPlanPrecision(handle, CLFFT_SINGLE));
  CLFFT_CHECK(clfftSetPlanScale(handle, dir, scale));
  CLFFT_CHECK(clfftSetPlanBatchSize(handle, batch_size));
  CLFFT_CHECK(clfftSetPlanDistance(handle, idist, odist));
  CLFFT_CHECK(clfftSetPlanInStride(handle, CLFFT_2D, instrides));
  CLFFT_CHECK(clfftSetPlanOutStride(handle, CLFFT_2D, outstrides));
  CLFFT_CHECK(clfftBakePlan(handle, 1,
              const_cast<cl_command_queue *>(&(queue.handle().get())),
              NULL, NULL));

  return handle;
}

clfftPlanHandle ClFFTState::createInPlaceManyPlanHandle(int height, int width,
    int batch_size, clfftDirection dir) {
  if (!initialized_) {
    LOG(INFO) << "clfft does not setup.";
    return (clfftPlanHandle)NULL;
  }
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();
  viennacl::ocl::command_queue &queue = ctx.get_queue();

  clfftPlanHandle handle;
  float scale = 1.f;
  int idist, odist;
  size_t instrides[2], outstrides[2];
  size_t lengths[2] = { (size_t)width, (size_t)height };
  CLFFT_CHECK(clfftCreateDefaultPlan(&handle, ctx.handle().get(),
              CLFFT_2D, lengths));

  if (CLFFT_FORWARD == dir) {  // FFT plan handle
    idist = height * 2*(width/2 + 1);
    odist = height * (width/2 + 1);
    instrides[0] = 1;
    instrides[1] = 2*(width/2 + 1);
    outstrides[0] = 1;
    outstrides[1] = (width/2 + 1);
    CLFFT_CHECK(clfftSetLayout(handle, CLFFT_REAL,
        CLFFT_HERMITIAN_INTERLEAVED));
  } else {
    CLFFT_CHECK(clfftDestroyPlan(&handle));
    LOG(ERROR) << "Not implemented";
    return (clfftPlanHandle)NULL;
  }

  CLFFT_CHECK(clfftSetResultLocation(handle, CLFFT_INPLACE));
  CLFFT_CHECK(clfftSetPlanPrecision(handle, CLFFT_SINGLE));
  CLFFT_CHECK(clfftSetPlanScale(handle, dir, scale));
  CLFFT_CHECK(clfftSetPlanBatchSize(handle, batch_size));
  CLFFT_CHECK(clfftSetPlanInStride(handle, CLFFT_2D, instrides));
  CLFFT_CHECK(clfftSetPlanOutStride(handle, CLFFT_2D, outstrides));
  CLFFT_CHECK(clfftSetPlanDistance(handle, idist, odist));
  CLFFT_CHECK(clfftBakePlan(handle, 1,
    const_cast<cl_command_queue*>(&(queue.handle().get())), NULL, NULL));

  return handle;
}

}  // namespace caffe
#endif  // USE_GREENTEA && USE_FFT
#endif  // CPU_ONLY
