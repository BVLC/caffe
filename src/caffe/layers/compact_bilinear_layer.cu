#include <algorithm>
#include <vector>

#include "caffe/layers/compact_bilinear_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

#define CHECK_CUFFT(X) CHECK_EQ((X), CUFFT_SUCCESS)

// overloaded functions, to support float and double
cublasStatus_t cublasgeam(cublasHandle_t handle, cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, const float *alpha,
        const float *A, int lda, const float *beta, const float *B, int ldb,
        float *C, int ldc) {
    return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B,
            ldb, C, ldc);
}
cublasStatus_t cublasgeam(cublasHandle_t handle, cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, const double *alpha,
        const double *A, int lda, const double *beta, const double *B, int ldb,
        double *C, int ldc) {
    return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B,
            ldb, C, ldc);
}
// caffe wrapper of transpose function
// dst=src^T, with the src size being M*N
template<typename Dtype>
void caffe_gpu_transpose(int M, const int N, const Dtype* src, Dtype* dst) {
    CHECK(src != dst) << "support out of place transpose only";
    Dtype alpha = 1.0, beta = 0.0;
    CHECK_EQ(
            cublasgeam(Caffe::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, M, N,
                    &alpha, src, N, &beta, dst, M, dst, M),
            CUBLAS_STATUS_SUCCESS);
}

template<typename Dtype>
void transpose_batch(const int batchlen, const int M, const int N,
        const Dtype* src, Dtype* dst) {
    const int step = M * N;
    for (int ins = 0; ins < batchlen; ++ins)
        caffe_gpu_transpose(M, N, src + ins * step, dst + ins * step);
}

// wrappers to deal with atomic add of double
__device__ void caffe_atomic_add(float* dst, float val) {
    atomicAdd(dst, val);
}
__device__ void caffe_atomic_add(double* address, double val) {
    // code example in the official document at:
    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
    //      #atomic-functions

    // NOLINT_NEXT_LINE(runtime/int)
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    // NOLINT_NEXT_LINE(runtime/int)
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN
        // (since NaN != NaN)
    } while (assumed != old);
}
// do the getCount and do transpose along the way
// should clear top to 0 before call
template<typename Dtype>
__global__ void GPUCountAndTranspose(const int nthreads, const int * hh,
        const Dtype * ss, const Dtype* bottom, Dtype* top, const int hw,
        const int C, const int num_output_) {
    // input batchlen*C*hw
    // output batchlen*hw*num_output, the transpose of the original output
    CUDA_KERNEL_LOOP(index, nthreads) {
        // nthreads is the total number of things you need to do
        // index is the current INPUT point to be computed
        int left = index % (C * hw);
        const int ibatch = index / (C * hw);
        const int ic = left / hw;
        const int ihw = left % hw;
        // get the target location
        const int target = ibatch * (hw * num_output_) + ihw * num_output_
                + hh[ic];
        // atomic add only supports float not double
        caffe_atomic_add(top + target, ss[ic] * bottom[index]);
    }
}

// some wrappers around cufftExec
// float forward
cufftResult cufftExec(cufftHandle plan, const float *idata,
        CaffeComplex<float> *odata) {
    return cufftExecR2C(plan,
            reinterpret_cast<cufftReal*>(const_cast<float*>(idata)),
            reinterpret_cast<cufftComplex*>(odata));
}
// double forward
cufftResult cufftExec(cufftHandle plan, const double *idata,
        CaffeComplex<double> *odata) {
    return cufftExecD2Z(plan,
            reinterpret_cast<cufftDoubleReal*>(const_cast<double*>(idata)),
            reinterpret_cast<cufftDoubleComplex*>(odata));
}
// float inverse
cufftResult cufftExec(cufftHandle plan, const CaffeComplex<float> *idata,
        float *odata) {
    return cufftExecC2R(plan,
            reinterpret_cast<cufftComplex*>(
                    const_cast<CaffeComplex<float>*>(idata)),
            reinterpret_cast<cufftReal*>(odata));
}
// double inverse
cufftResult cufftExec(cufftHandle plan, const CaffeComplex<double> *idata,
        double *odata) {
    return cufftExecZ2D(plan,
            reinterpret_cast<cufftDoubleComplex*>(
                    const_cast<CaffeComplex<double>*>(idata)),
            reinterpret_cast<cufftDoubleReal*>(odata));
}
// call cufft to do batch*nffts
// cufftReal* src; cufftComplex *output
template<typename Dtype>
void CompactBilinearLayer<Dtype>::caffe_gpu_fft(const int batchlen,
        const int hw, const int nfft, const Dtype* src,
        CaffeComplex<Dtype>* output) {
    if (batchlen == batchsz) {
        CHECK_CUFFT(cufftExec(plan_noinv_batch, src, output));
    } else {
        const int step_in = hw * nfft;
        const int step_out = hw * (floor(1.0 * nfft / 2) + 1);
        for (int i = 0; i < batchlen; ++i) {
            CHECK_CUFFT(
                    cufftExec(plan_noinv_1, src + step_in * i,
                            output + step_out * i));
        }
    }
}

template<typename Dtype>
void CompactBilinearLayer<Dtype>::caffe_gpu_ifft(const int batchlen,
        const int hw, const int nfft, const CaffeComplex<Dtype>* src,
        Dtype* output) {
    if (batchlen == batchsz) {
        CHECK_CUFFT(cufftExec(plan_inv_batch, src, output));
    } else {
        const int step_in = hw * (floor(1.0 * nfft / 2) + 1);
        const int step_out = hw * nfft;
        for (int i = 0; i < batchlen; ++i) {
            CHECK_CUFFT(
                    cufftExec(plan_inv_1, src + step_in * i,
                            output + step_out * i));
        }
    }
}

// Complex multiplication
template<typename Dtype>
static __device__ __host__ inline CaffeComplex<Dtype> ComplexMul(
        const CaffeComplex<Dtype> &a, const CaffeComplex<Dtype> &b) {
    CaffeComplex<Dtype> c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// entrywise multiplication: y[i]=a[i]*b[i]
template<typename Dtype>
__global__ void complexMul(const int nthreads, const CaffeComplex<Dtype>* a,
        const CaffeComplex<Dtype>* b, CaffeComplex<Dtype>* y) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // nthreads is the total number of entries
        y[index] = ComplexMul(a[index], b[index]);
    }
}

// dispatchers
cublasStatus_t cublasgemv(cublasHandle_t handle, cublasOperation_t trans, int m,
        int n, const float *alpha, const float *A, int lda, const float *x,
        int incx, const float *beta, float *y, int incy) {
    return
    cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
cublasStatus_t cublasgemv(cublasHandle_t handle, cublasOperation_t trans, int m,
        int n, const double *alpha, const double *A, int lda, const double *x,
        int incx, const double *beta, double *y, int incy) {
    return
    cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
// sum the columns of a M*N source matrix and store it to dst
template<typename Dtype>
void caffe_sum_cols(const int M, const int N, const Dtype* src, Dtype* dst,
        Dtype* ones_hw) {
    Dtype alpha = 1.0, beta = 0.0;
    CHECK_EQ(
            cublasgemv(Caffe::cublas_handle(), CUBLAS_OP_T, N, M, &alpha, src,
                    N, ones_hw, 1, &beta, dst, 1), CUBLAS_STATUS_SUCCESS);
}

template<>
void CompactBilinearLayer<float>::Initializations(const int hw) {
    int n = num_output_;
    // each plan is signatured by (R2C, batchsz)
    CHECK_CUFFT(cufftPlanMany(&plan_noinv_batch, 1, &n, NULL, 0, 0, NULL, 0, 0,
            CUFFT_R2C, batchsz*hw));
    CHECK_CUFFT(cufftPlanMany(&plan_noinv_1 , 1, &n, NULL, 0, 0, NULL, 0, 0,
            CUFFT_R2C, hw));
    CHECK_CUFFT(cufftPlanMany(&plan_inv_batch, 1, &n, NULL, 0, 0, NULL, 0, 0,
            CUFFT_C2R, batchsz*hw));
    CHECK_CUFFT(cufftPlanMany(&plan_inv_1 , 1, &n, NULL, 0, 0, NULL, 0, 0,
            CUFFT_C2R, hw));
}

template<>
void CompactBilinearLayer<double>::Initializations(const int hw) {
    int n = num_output_;
    // each plan is signatured by (R2C, batchsz)
    CHECK_CUFFT(cufftPlanMany(&plan_noinv_batch, 1, &n, NULL, 0, 0, NULL, 0, 0,
            CUFFT_D2Z, batchsz*hw));
    CHECK_CUFFT(cufftPlanMany(&plan_noinv_1 , 1, &n, NULL, 0, 0, NULL, 0, 0,
            CUFFT_D2Z, hw));
    CHECK_CUFFT(cufftPlanMany(&plan_inv_batch, 1, &n, NULL, 0, 0, NULL, 0, 0,
            CUFFT_Z2D, batchsz*hw));
    CHECK_CUFFT(cufftPlanMany(&plan_inv_1 , 1, &n, NULL, 0, 0, NULL, 0, 0,
            CUFFT_Z2D, hw));
}

template<typename Dtype>
void CompactBilinearLayer<Dtype>::Forward_gpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const int hw = bottom[0]->count(2);
    if (!plan_init) {
        // some init commands that will only be executed once
        plan_init = true;
        Initializations(hw);
        // get an all one vector
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ones_hw),
                              sizeof(Dtype) * hw));
        caffe_gpu_set(hw, Dtype(1.0), ones_hw);
    }

    // memory pointer short hand
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* bottom_data[2] =
            { bottom[0]->gpu_data(), bottom[1]->gpu_data() };

    const int step_top = top[0]->count(1);
    const int step_bottom[2] = { bottom[0]->count(1), bottom[1]->count(1) };

    const int C[2] = { bottom[0]->shape(1), bottom[1]->shape(1) };

    // temporary space allocation
    Dtype* batchSpace[2];
    CaffeComplex<Dtype>* fftSpace[2];
    for (int ipoly = 0; ipoly < 2; ++ipoly) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&batchSpace[ipoly]),
                        batchsz * num_output_ * hw * sizeof(Dtype)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fftSpace[ipoly]),
               batchsz * num_complex_out * hw * sizeof(CaffeComplex<Dtype>)));
    }

    // batching process each bottom
    const int totalSamples = bottom[0]->shape(0);
    for (int batchStart = 0; batchStart < totalSamples; batchStart += batchsz) {
        const int batchlen = min(batchsz, totalSamples - batchStart);

        for (int ipoly = 0; ipoly < 2; ++ipoly) {
            // some short hands
            Dtype* space = batchSpace[ipoly];
            const int * hh = randh_[ipoly].gpu_data();
            const Dtype * ss = rands_[ipoly].gpu_data();
            int nthreads;

            caffe_gpu_set(batchlen * hw * num_output_, Dtype(0.0), space);
            // first get count and transpose
            nthreads = batchlen * step_bottom[ipoly];
            GPUCountAndTranspose<Dtype>
            // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
                    nthreads, hh, ss,
                    bottom_data[ipoly] + batchStart * step_bottom[ipoly],
                    space, hw, C[ipoly], num_output_);
            // now space is batchlen*hw*num_output

            // then do FFT
            caffe_gpu_fft(batchlen, hw, num_output_, space, fftSpace[ipoly]);
        }
        // entry-wise multiplication
        int nthreads = batchlen * hw * num_complex_out;
        complexMul<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
                nthreads, fftSpace[0], fftSpace[1], fftSpace[0]);

        // ifft
        caffe_gpu_ifft(batchlen, hw, num_output_, fftSpace[0], batchSpace[0]);

        // transpose back
        Dtype* out_target;
        if (sum_pool_)
            out_target = batchSpace[1];
        else
            out_target = top_data + batchStart * step_top;

        transpose_batch(batchlen, hw, num_output_, batchSpace[0], out_target);

        if (sum_pool_)
            caffe_sum_cols(batchlen * num_output_, hw, out_target,
                    top_data + batchStart * step_top, ones_hw);
    }

    // temporary space destroy
    for (int ipoly = 0; ipoly < 2; ++ipoly) {
        CUDA_CHECK(cudaFree(batchSpace[ipoly]));
        CUDA_CHECK(cudaFree(fftSpace[ipoly]));
    }
}

template<typename Dtype>
__global__ void copy_and_transpose(const int nthreads, const int batch,
        const int num_output_, const int hw, const Dtype* src, Dtype* dst) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // src size: batch*num_output_
        // dst size: batch*hw*num_output_
        // index over dst
        const int left = index % (hw * num_output_);
        const int ibatch = index / (hw * num_output_);
        const int ihw = left / num_output_;
        const int iout = left % num_output_;

        dst[index] = src[ibatch * num_output_ + iout];
    }
}

// C, dst, hh and ss are complement
template<typename Dtype>
__global__ void assign_back(const int nthreads, const Dtype* src, Dtype* dst,
        const int* hh, const Dtype* ss, const int batchlen, const int C,
        const int hw, const int num_output_) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // src size: batchlen*hw*num_output
        // dst size: batchlen*C*hw
        // index over dst
        const int left = index % (hw * C);
        const int ibatch = index / (hw * C);
        const int ic = left / hw;
        const int ihw = left % hw;

        dst[index] += ss[ic] * src[(ibatch * hw + ihw) * num_output_ + hh[ic]];
    }
}

template<typename Dtype>
__device__ void caffe_gpu_swap(Dtype* a, Dtype* b) {
    if (a == b)
        return;
    Dtype t = *a;
    *a = *b;
    *b = t;
}

template<typename Dtype>
__global__ void fliplr(const int nthreads, Dtype* src, const int M,
        const int N) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // src & dst are M*N
        // flip left right, loop over src
        const int m = index / N;
        const int n = index % N;
        if ((n <= (N / 2)) && (n >= 1))
            caffe_gpu_swap(src + index, src + index - n + N - n);
    }
}

template<typename Dtype>
void CompactBilinearLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
    if ((!propagate_down[0]) && (!propagate_down[1]))
        return;
    // process the same bottom case
    // when the two bottoms are the same, one propagate down requires the other
    vector<bool> pd = propagate_down;
    if (bottom[0] == bottom[1])
        pd[0] = pd[1] = true;

    // memory pointer short hand
    const Dtype* bottom_data[2] =
            { bottom[0]->gpu_data(), bottom[1]->gpu_data() };
    Dtype* bottom_diff[2] = { bottom[0]->mutable_gpu_diff(), bottom[1]
            ->mutable_gpu_diff() };
    for (int i = 0; i < 2; ++i)
        caffe_gpu_set(bottom[i]->count(), Dtype(0.0), bottom_diff[i]);
    const Dtype* top_diff = top[0]->gpu_diff();

    const int step_bottom[2] = { bottom[0]->count(1), bottom[1]->count(1) };
    const int step_top = top[0]->count(1);

    const int C[2] = { bottom[0]->shape(1), bottom[1]->shape(1) };
    const int hw = bottom[0]->count(2);

    // the pointer to the (repeated) derivative
    Dtype* dzdy;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dzdy),
            batchsz * num_output_ * hw * sizeof(Dtype)));
    // fft[0] for derivative, fft[1] for data
    CaffeComplex<Dtype>* fftSpace[2];
    for (int ipoly = 0; ipoly < 2; ++ipoly)
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fftSpace[ipoly]),
            batchsz * num_complex_out * hw * sizeof(CaffeComplex<Dtype> )));

    // batching process each bottom
    const int totalSamples = bottom[0]->shape(0);
    for (int batchStart = 0; batchStart < totalSamples; batchStart += batchsz) {
        const int batchlen = min(batchsz, totalSamples - batchStart);
        // (copy and) transpose the derivative
        if (sum_pool_) {
            int nthreads = batchlen * hw * num_output_;
            // copy and transpose the derivative
        copy_and_transpose<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
                nthreads, batchlen, num_output_, hw,
                top_diff + batchStart * step_top, dzdy);
    } else {
        // transpose the derivative
        transpose_batch<Dtype>(batchlen, num_output_, hw,
                top_diff + batchStart * step_top, dzdy);
    }

    // fft the derivative, stored in fftSpace[0]
    caffe_gpu_fft(batchlen, hw, num_output_, dzdy, fftSpace[0]);

    for (int ipoly = 0; ipoly < 2; ++ipoly)
        if (pd[1 - ipoly]) {
            // some short hands
            const int * hh = randh_[ipoly].gpu_data();
            const Dtype * ss = rands_[ipoly].gpu_data();
            int nthreads;

            // first get count and transpose, reuse the dzdy space
            nthreads = batchlen * step_bottom[ipoly];
            caffe_gpu_set(batchlen * hw * num_output_, Dtype(0.0), dzdy);
            GPUCountAndTranspose<Dtype>
            // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
                    nthreads, hh, ss,
                    bottom_data[ipoly] + batchStart * step_bottom[ipoly],
                    dzdy, hw, C[ipoly], num_output_);
            // now dzdy is batchlen*hw*num_output_

            // fliplr(:, 2:end)
            nthreads = batchlen * hw * num_output_;
            fliplr<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
                    nthreads, dzdy, batchlen * hw, num_output_);

            // fft data
            caffe_gpu_fft(batchlen, hw, num_output_, dzdy, fftSpace[1]);

            // elementwise mul
            nthreads = batchlen * hw * num_complex_out;
            complexMul<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
                    nthreads, fftSpace[0], fftSpace[1], fftSpace[1]);

            // ifft, again reuse dzdy
            caffe_gpu_ifft(batchlen, hw, num_output_, fftSpace[1], dzdy);

            // complement projection
            nthreads = batchlen * hw * C[1 - ipoly];
        assign_back<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
                nthreads, dzdy,
                bottom_diff[1-ipoly] + batchStart * step_bottom[1-ipoly],
                randh_[1-ipoly].gpu_data(), rands_[1-ipoly].gpu_data(),
                batchlen, C[1-ipoly], hw, num_output_);
    }
}

// temporary space destroy
CUDA_CHECK(cudaFree(dzdy));
CUDA_CHECK(cudaFree(fftSpace[0]));
CUDA_CHECK(cudaFree(fftSpace[1]));
}

INSTANTIATE_LAYER_GPU_FUNCS(CompactBilinearLayer);

}  // namespace caffe
