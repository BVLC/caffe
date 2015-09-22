#ifdef USE_AUDIO
#include "caffe/util/fft.hpp"

#include <cufft.h>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
struct FastFourierTransformPImpl {
    cufftHandle plan;
};

template <>
FastFourierTransform_gpu<double>::FastFourierTransform_gpu(int packetSize)
: _log2Size(std::ceil(std::log(packetSize) / std::log(2))),
_packetSize(static_cast<int>(std::pow(2, _log2Size))),
_pimpl(new FastFourierTransformPImpl<double>())
{}

template <>
FastFourierTransform_gpu<float>::FastFourierTransform_gpu(int packetSize)
: _log2Size(std::ceil(std::log(packetSize) / std::log(2))),
_packetSize(static_cast<int>(std::pow(2, _log2Size))),
_pimpl(new FastFourierTransformPImpl<float>())
{}

template <typename Dtype>
FastFourierTransform_gpu<Dtype>::~FastFourierTransform_gpu() {}

template <>
int FastFourierTransform_gpu<double>::process(double* input_data,
                                              double* output_data, int size) {
  CHECK_EQ(size, _packetSize);

  if (output_data) {
    caffe_copy(size, input_data, output_data);
  } else {
    output_data = input_data;
  }

  CHECK_EQ(cufftPlan1d(&(_pimpl->plan), size, CUFFT_D2Z, 1), CUFFT_SUCCESS)
                                                << "Creation of plan failed.";
  CHECK_EQ(cufftExecD2Z(_pimpl->plan,
                        reinterpret_cast<cufftDoubleReal*>(output_data),
                        reinterpret_cast<cufftDoubleComplex*>(output_data)),
                        CUFFT_SUCCESS) << "Execution of cuFFT failed.";
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess)
                                              << "CUDA failed to synchronize.";
  CHECK_EQ(cufftDestroy(_pimpl->plan), CUFFT_SUCCESS)
                                              << "Failed to destroy cuFFT.";
  caffe_cvnrm(reinterpret_cast<std::complex<double>*>(output_data),
                       output_data, size);
  caffe_scal(size, 1.0 / size, output_data);

  return size;
}

template <>
int FastFourierTransform_gpu<float>::process(float* input_data,
                                                float* output_data, int size) {
  CHECK_EQ(size, _packetSize);

  if (output_data) {
    caffe_copy(size, input_data, output_data);
  } else {
    output_data = input_data;
  }

  CHECK_EQ(cufftPlan1d(&(_pimpl->plan), size, CUFFT_R2C, 1), CUFFT_SUCCESS)
                                << "Creation of plan failed.";
  CHECK_EQ(cufftExecR2C(_pimpl->plan, reinterpret_cast<cufftReal*>(output_data),
                                reinterpret_cast<cufftComplex*>(output_data)),
                                CUFFT_SUCCESS) << "Execution of cuFFT failed.";
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess)
                                << "CUDA failed to synchronize.";
  CHECK_EQ(cufftDestroy(_pimpl->plan), CUFFT_SUCCESS)
                                << "Failed to destroy cuFFT.";
  caffe_cvnrm(reinterpret_cast<std::complex<float>*>(output_data),
                     output_data, size);
  caffe_scal(size, 1.0f / size, output_data);

  return size;
}

INSTANTIATE_CLASS(FastFourierTransform_gpu);

}  // namespace caffe
#endif  // USE_AUDIO
