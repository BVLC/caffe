
#ifndef CAFFE_UTIL_caffe_cpu_fft_H_
#define CAFFE_UTIL_caffe_cpu_fft_H_
#ifdef USE_FFT

#ifndef CPU_ONLY
#include <cufft.h>
#endif

#include <fftw3.h>

#include <complex>


namespace caffe {

template <typename Dtype>
void* caffe_cpu_fft_malloc(int n);

template <typename Dtype>
void  caffe_cpu_fft_free(void* p);

template <typename Dtype>
void* caffe_cpu_fft_plan_dft_r2c_2d(int n0, int n1,
    Dtype *in, std::complex<Dtype> *out, unsigned flags);

template <typename Dtype>
void* caffe_cpu_fft_plan_dft_c2r_2d(int n0, int n1,
    std::complex<Dtype> *in, Dtype *out, unsigned flags);

template <typename Dtype>
void* caffe_cpu_fft_plan_many_dft_r2c(int rank, const int *n, int howmany,
    Dtype *in, const int *inemded, int istride, int idist,
    std::complex<Dtype> *out, const int *onembed, int ostride, int odist,
    unsigned flags);

template <typename Dtype>
void caffe_cpu_fft_destroy_plan(void* plan);

template <typename Dtype>
void caffe_cpu_fft_execute(const void* plan);

template <typename Dtype>
void caffe_cpu_fft_execute_dft_r2c(const void* plan,
    Dtype *in, std::complex<Dtype> *out);

template <typename Dtype>
void caffe_cpu_fft_execute_dft_c2r(const void* plan,
    std::complex<Dtype> *in, Dtype  *out);

inline
int caffe_cpu_fft_init_threads(void ) {
  return(fftw_init_threads());
}

inline
void  caffe_cpu_fft_plan_with_nthreads(int num_of_threads_) {
  fftw_plan_with_nthreads(num_of_threads_);
}

inline
void caffe_cpu_fft_cleanup_threads(void) {
  fftw_cleanup_threads();
}

// --- gpu-------------------------------------------------

#ifndef CPU_ONLY
template <typename Dtype>
void fft_gpu_copy2buffer(Dtype* fft_gpu_weights_real_, const Dtype* weight,
    int num_output, int group, int channels, int ker_h_, int ker_w_,
    int fft_height, int fft_width);

template <typename Dtype>
void fft_gpu_copy2buffer2D_in(Dtype* map_out, const Dtype* map_in,
    int width_out, int height, int width, int stride_h_,
    int stride_w_, int pad_h_, int pad_w_, Dtype units);

template <typename Dtype>
void fft_gpu_copy2buffer2D_out(Dtype* map_out, const Dtype* map_in,
    int height_out, int width_out, int fft_height, int fft_width,
     int stride_h_, int stride_w_, int pad_h_, int pad_w_, Dtype units);

template <typename Dtype>
void caffe_gpu_elementMulConj(std::complex<Dtype>* dst,
    std::complex<Dtype>* src1, std::complex<Dtype>* src2, int size);

template <typename Dtype>
void caffe_gpu_elementMul(std::complex<Dtype>* dst,
    std::complex<Dtype>* src1, std::complex<Dtype>* src2, int size);

template <typename Dtype>
void caffe_gpu_fft_execute_dft_r2c(cufftHandle plan,
    Dtype *in, std::complex<Dtype> *out);

template <typename Dtype>
void caffe_gpu_fft_execute_dft_r2c_inplace(cufftHandle plan,
    std::complex<Dtype> *inout);

template <typename Dtype>
void caffe_gpu_fft_execute_dft_c2r(cufftHandle plan,
    std::complex<Dtype> *in, Dtype  *out);
#endif

}  // namespace caffe

#endif // USE_FFT

#endif  // CAFFE_UTIL_caffe_cpu_fft_H_
