#ifdef USE_AUDIO
#include "caffe/util/fft.hpp"

#include <fftw3.h>

#include <cmath>
#include <valarray>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void hammingWindow(Dtype* data, int size);

template <typename Dtype>
struct FastFourierTransformPImpl {
  explicit FastFourierTransformPImpl(int size) : buffer(size + 2),
                                                 window(size) {
    hammingWindow(&window[0], size);
  }
  std::valarray<Dtype> buffer;
  std::valarray<Dtype> window;
};

template <>
FastFourierTransform_cpu<double>::FastFourierTransform_cpu(int packetSize)
: _log2Size(std::ceil(std::log(packetSize) / std::log(2))),
_packetSize(static_cast<int>(std::pow(2, _log2Size))),
_pimpl(new FastFourierTransformPImpl<double>(packetSize))
{}

template <>
FastFourierTransform_cpu<float>::FastFourierTransform_cpu(int packetSize)
: _log2Size(std::ceil(std::log(packetSize) / std::log(2))),
_packetSize(static_cast<int>(std::pow(2, _log2Size))),
_pimpl(new FastFourierTransformPImpl<float>(packetSize))
{}

template <typename Dtype>
FastFourierTransform_cpu<Dtype>::~FastFourierTransform_cpu() {}

template <>
int FastFourierTransform_cpu<double>::process(double* input_data,
                                          double* output_data, int size) {
  CHECK_LE(size, _packetSize);

  // Apply window to data
  caffe_copy(size, input_data, &_pimpl->buffer[0]);
  caffe_mul(size, &_pimpl->buffer[0], &_pimpl->window[0],
            &_pimpl->buffer[0]);

  fftw_plan plan = fftw_plan_dft_r2c_1d(size, &_pimpl->buffer[0],
                    reinterpret_cast<fftw_complex*>(&_pimpl->buffer[0]),
                    FFTW_ESTIMATE);
  CHECK(plan) << "Could not create FFT plan.";
  fftw_execute(plan);
  fftw_destroy_plan(plan);

  // Normalize data
  caffe_cvnrm(reinterpret_cast<std::complex<double>*>(&_pimpl->buffer[0]),
              &_pimpl->buffer[0], size / 2);
  caffe_scal(size, 1.0 / size, &_pimpl->buffer[0]);

  if (output_data) {
    caffe_copy(size / 2, &_pimpl->buffer[0], output_data);
  } else {
    caffe_copy(size / 2, &_pimpl->buffer[0], input_data);
  }

  return size;
}

template <>
int FastFourierTransform_cpu<float>::process(float* input_data,
                                            float* output_data, int size) {
  CHECK_LE(size, _packetSize);

  // Apply window to data
  caffe_copy(size, input_data, &_pimpl->buffer[0]);
  caffe_mul(size, &_pimpl->buffer[0], &_pimpl->window[0],
            &_pimpl->buffer[0]);

  fftwf_plan plan = fftwf_plan_dft_r2c_1d(size, &_pimpl->buffer[0],
                   reinterpret_cast<fftwf_complex*>(&_pimpl->buffer[0]),
                   FFTW_ESTIMATE);
  CHECK(plan) << "Could not create FFT plan.";
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);

  // Normalize data
  caffe_cvnrm(reinterpret_cast<std::complex<float>*>(&_pimpl->buffer[0]),
              &_pimpl->buffer[0], size / 2);
  caffe_scal(size, 1.0f / size, &_pimpl->buffer[0]);

  if (output_data) {
    caffe_copy(size / 2, &_pimpl->buffer[0], output_data);
  } else {
    caffe_copy(size / 2, &_pimpl->buffer[0], input_data);
  }

  return size;
}

template <>
void hammingWindow(float* data, int size) {
  const float alpha = 0.54;
  const float beta = 0.46;

  for (int i = 0; i < size; ++i) {
      data[i] = alpha - (beta * (2 * M_PI * i / (size - 1)));
  }
}

template <>
void hammingWindow(double* data, int size) {
  const double alpha = 0.54;
  const double beta = 0.46;

  for (int i = 0; i < size; ++i) {
      data[i] = alpha - (beta * (2 * M_PI * i / (size - 1)));
  }
}

INSTANTIATE_CLASS(FastFourierTransform_cpu);

}  // namespace caffe
#endif  // USE_AUDIO
