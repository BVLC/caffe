#ifndef CAFFE_UTIL_CL_FFT_HELPER_H_
#define CAFFE_UTIL_CL_FFT_HELPER_H_
#ifdef CMAKE_BUILD
#include <caffe_config.h>
#endif
#ifndef CPU_ONLY
#if defined(USE_GREENTEA) && defined(USE_FFT)
#include <clFFT.h>
#include <map>
#include <utility>

namespace caffe {

typedef std::pair<int, int>FFTSize;
typedef std::pair<FFTSize, int> KeyType;
typedef std::pair<KeyType, clfftPlanHandle> KeyType_HandlePtr;

class ClFFTState {
 public:
  ClFFTState();
  void setup();
  void teardown();
  clfftPlanHandle getForwardInPlaceFFTManyPlanHandle(const int height,
      const int width, int batch_size);
  clfftPlanHandle getForwardOutOfPlaceFFTManyPlanHandle(const int height,
      const int width, int batch_size);
  clfftPlanHandle getBackwardOutOfPlaceFFTManyPlanHandle(const int height,
      const int width, int batch_size);
  clfftPlanHandle getForwardOutOfPlaceIFFTManyPlanHandle(const int height,
      const int width, int batch_size);
  clfftPlanHandle getBackwardOutOfPlaceIFFTManyPlanHandle(const int height,
      const int width, int batch_size);

 private:
  // Support only Forward and Backward, otherwise return Not implemented
  clfftPlanHandle createOutOfPlaceManyPlanHandle(int height, int width,
      int batch_size, clfftDirection dir = CLFFT_FORWARD);
  // Support only Forward, otherwise return Not implemented
  clfftPlanHandle createInPlaceManyPlanHandle(int height, int width,
      int batch_size, clfftDirection dir = CLFFT_FORWARD);

 private:
  bool initialized_;
  std::map<KeyType, clfftPlanHandle> forward_fft_inplace_many_handle_map_;
  std::map<KeyType, clfftPlanHandle> forward_fft_many_handle_map_;
  std::map<KeyType, clfftPlanHandle> backward_fft_many_handle_map_;
  std::map<KeyType, clfftPlanHandle> forward_ifft_many_handle_map_;
  std::map<KeyType, clfftPlanHandle> backward_ifft_many_handle_map_;
};

}  // namespace caffe

#endif  // USE_GREENTEA && USE_FFT
#endif  // CPU_ONLY
#endif  // CAFFE_UTIL_CL_FFT_HELPER_H_

