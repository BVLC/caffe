#include <stdlib.h>

#include <algorithm>
#include <vector>

#include "caffe/layers/compact_bilinear_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template<typename Dtype>
CompactBilinearLayer<Dtype>::CompactBilinearLayer(const LayerParameter& param) : Layer<Dtype>(param) {
    fft_cfg_noinv = NULL;
    fft_cfg_inverse = NULL;
    buf_float = NULL;
}

template<typename Dtype>
void CompactBilinearLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // read in the parameters
    const CompactBilinearParameter& compact_param = this->layer_param_
            .compact_bilinear_param();
    CHECK(compact_param.has_num_output()) << "num_output should be specified";
    num_output_ = static_cast<int>(compact_param.num_output());
    CHECK_GT(num_output_, 0) << "num_output should be positive.";
    num_complex_out = floor(1.0 * num_output_ / 2) + 1;
    sum_pool_ = compact_param.sum_pool();

    // the fft plans for cpu execution
    fft_cfg_noinv = kiss_fftr_alloc(num_output_, 0, NULL, NULL);
    fft_cfg_inverse = kiss_fftr_alloc(num_output_, 1, NULL, NULL);
    // a temporary space for double fft
    buf_float = reinterpret_cast<float*>(malloc(num_output_ * sizeof(float)));

#ifndef CPU_ONLY
    plan_init = false;
    // we have tried batchsz > 1, it doesn't help with speed.
    // to be maximally compatible with small memory GPU, set it to 1.
    batchsz = 1;
#endif
}

template<typename Dtype>
void randi(const int n, const int maxv, Dtype* r) {
    // generate n values, each of them in [0, maxv), and store them in r[i]
    for (int i = 0; i < n; ++i)
        // Since random seed is set in the Reshape function to fix the seed
        // We don't want to change caffe's random seed by inserting this
        // layer. Thus we resort to C's default random number generator.
        // NOLINT_NEXT_LINE(caffe/random_fn)
        r[i] = rand() % maxv;
}

template<typename Dtype>
void CompactBilinearLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // check the shape of two bottom compatible
    for (int i = 0; i < 2; ++i) {
        const int num_axes = bottom[i]->num_axes();
        CHECK_EQ(num_axes, 4) << "Bilinear layer only support 4 dim blobs.";
    }
    for (int axis = 0; axis < 4; ++axis) {
        // the number of channels could be different
        if (axis == 1) {
            continue;
        }
        CHECK_EQ(bottom[0]->shape(axis), bottom[1]->shape(axis))
                << "Two bottom blobs not compatible at axis " << axis << ".";
    }
    CHECK((bottom[0] != top[0]) && (bottom[1] != top[0]))
            << "could not do in place operation.";

    // then assign the shape of the top blob
    vector<int> top_shape = bottom[0]->shape();
    top_shape[1] = num_output_;
    if (sum_pool_)
        top_shape[2] = top_shape[3] = 1;
    top[0]->Reshape(top_shape);

    if (randh_[0].count() == 0) {
        // generate the random weights, this should be only executed once.
        const int C[2] = { bottom[0]->shape(1), bottom[1]->shape(1) };
        // use default C random number generator to fix randomness
        srand(1);
        for (int i = 0; i < 2; ++i) {
            // fill randh_
            randh_[i].Reshape(vector<int>(1, C[i]));
            int* hdata = randh_[i].mutable_cpu_data();
            randi(C[i], num_output_, hdata);

            // fill rands_
            rands_[i].Reshape(vector<int>(1, C[i]));
            Dtype* sdata = rands_[i].mutable_cpu_data();
            randi(C[i], 2, sdata);
            // and transform it from [0, 1] to [-1, 1]
            for (int j = 0; j < C[i]; ++j)
                sdata[j] = 2 * sdata[j] - 1;
        }
    }
}

template<typename Dtype>
CompactBilinearLayer<Dtype>::~CompactBilinearLayer() {
    // free the spaces
    if (fft_cfg_noinv) {
        kiss_fftr_free(fft_cfg_noinv);
    }
    if (fft_cfg_inverse) {
        kiss_fftr_free(fft_cfg_inverse);
    }
    if (buf_float) {
        free(buf_float);
    }
    // TODO: don't know how to call GPU clean functions from here.
    /*
     CHECK_EQ(cufftDestroy(plan_noinv_batch), CUFFT_SUCCESS);
     CHECK_EQ(cufftDestroy(plan_noinv_1), 	 CUFFT_SUCCESS);
     CHECK_EQ(cufftDestroy(plan_inv_batch),   CUFFT_SUCCESS);
     CHECK_EQ(cufftDestroy(plan_inv_1), 		 CUFFT_SUCCESS);

     CUDA_CHECK(cudaFree(ones_hw));
     */
}

// Notes on current CPU FFT library - kiss_fft:
// 1. this lib support float (no double), due to the nature of kiss_fft.
//    This is less flexible, but probably not a big problem
// 2. no batch fft transform. Only one at a time. (Memory access slow)
// 3. CPU version don't support odd number num_output_, due to kiss_fftr
// 4. didn't use SSE or AVX instructions. (theoretically 8x speed loss)
// 5. could potentially be replaced by FFTW or MKL, but that
//    will add more dependencies. (2x speed loss)
// 6. current kiss_fft lib operates at 2.2Gflops (5*N*log_2^N) on E5-2640
//    We bench-marked ffts, which claims to be the fastest lib (slightly
//    faster than MKL) on the same machine. With SSE enabled, it operates
//    at 6.0Gflops. Note that we didn't loop over a single copy of the
//    data, which will alleviate the cache problem. Instead, we simulate
//    the real world scenario, where each fft operates on different copies.

// wrappers around kiss_fft
void caffe_cpu_fft(kiss_fftr_cfg cfg, const float *timedata,
        kiss_fft_cpx *freqdata, float* buf_float, const int nfft) {
    kiss_fftr(cfg, (const kiss_fft_scalar *) timedata, freqdata);
}
void caffe_cpu_fft(kiss_fftr_cfg cfg, const double *timedata,
        kiss_fft_cpx *freqdata, float* buf_float, const int nfft) {
    for (int i = 0; i < nfft; ++i)
        buf_float[i] = timedata[i];
    kiss_fftr(cfg, (const kiss_fft_scalar *) buf_float, freqdata);
}
void caffe_cpu_ifft(kiss_fftr_cfg cfg, const kiss_fft_cpx *freqdata,
        float *timedata, float* buf_float, const int nfft) {
    kiss_fftri(cfg, freqdata, reinterpret_cast<kiss_fft_scalar *>(timedata));
}
void caffe_cpu_ifft(kiss_fftr_cfg cfg, const kiss_fft_cpx *freqdata,
        double *timedata, float* buf_float, const int nfft) {
    kiss_fftri(cfg, freqdata, reinterpret_cast<kiss_fft_scalar *>(buf_float));
    for (int i = 0; i < nfft; ++i)
        timedata[i] = buf_float[i];
}

template<typename Dtype>
void caffe_cpu_fft_batch(kiss_fftr_cfg fft_cfg_noinv, const Dtype* batchSpace,
        kiss_fft_cpx* fftSpace, int num_output_, float* buf_float, int hw) {
    int num_complex_out = floor(1.0 * num_output_ / 2) + 1;
    for (int loc = 0; loc < hw; ++loc)
        caffe_cpu_fft(fft_cfg_noinv, batchSpace + loc * num_output_,
                fftSpace + loc * num_complex_out, buf_float, num_output_);
}

template<typename Dtype>
void caffe_cpu_ifft_batch(kiss_fftr_cfg fft_cfg_inverse, Dtype* batchSpace,
        const kiss_fft_cpx* fftSpace, int num_output_, float* buf_float,
        int hw) {
    int num_complex_out = floor(1.0 * num_output_ / 2) + 1;
    for (int loc = 0; loc < hw; ++loc)
        caffe_cpu_ifft(fft_cfg_inverse, fftSpace + loc * num_complex_out,
                batchSpace + loc * num_output_, buf_float, num_output_);
}

template<typename Dtype>
void caffe_cpu_transpose(int M, int N, const Dtype* src, Dtype* dst) {
    const int sz = 16;
    // src is M*N, and dst is N*M
    for (int i = 0; i < M; i += sz)
        for (int j = 0; j < N; j += sz) {
            for (int i2 = i; i2 < std::min(M, i + sz); ++i2)
                for (int j2 = j; j2 < std::min(N, j + sz); ++j2)
                    dst[j2 * M + i2] = src[i2 * N + j2];
        }
}

template<typename Dtype>
void CompactBilinearLayer<Dtype>::CPUCountAndTranspose(const Blob<int>& randh,
        const Blob<Dtype>& rands, const Dtype* bottom, Dtype* top, const int hw,
        Dtype* temptop) {
    // randh & rands: vector of length C;
    // bottom: C*H*W; output: num_output_*H*W
    const int* h = randh.cpu_data();
    const Dtype* s = rands.cpu_data();
    caffe_set(hw * num_output_, Dtype(0.0), temptop);

    for (int c = 0; c < randh.count(); ++c) {
        const Dtype* bottom_c = bottom + c * hw;
        Dtype* top_c = temptop + h[c] * hw;
        caffe_cpu_axpby(hw, Dtype(s[c]), bottom_c, Dtype(1.0), top_c);
    }

    caffe_cpu_transpose(num_output_, hw, temptop, top);
}

// y[i]=complexMul(a[i], b[i])
template<typename Dtype>
void fft_cpx_mul(int n, const kiss_fft_cpx* a, kiss_fft_cpx* b,
        kiss_fft_cpx* y) {
    for (int i = 0; i < n; ++i) {
        Dtype outr = 0, outi = 0;
        outr = a[i].r * b[i].r - a[i].i * b[i].i;
        outi = a[i].r * b[i].i + a[i].i * b[i].r;
        y[i].r = outr;
        y[i].i = outi;
    }
}

template<typename Dtype>
void CompactBilinearLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(num_output_ % 2 , 0) <<
            "CPU version don't support odd num_output_";

    const Dtype* bottom_data[2] =
            { bottom[0]->cpu_data(), bottom[1]->cpu_data() };
    Dtype* top_data = top[0]->mutable_cpu_data();

    const int bottom_step[2] = { bottom[0]->count(1), bottom[1]->count(1) };
    const int top_step = top[0]->count(1);
    const int hw = bottom[0]->count(2);

    // generate the space for buffer
    Dtype* batchSpace[2] = {
        reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * num_output_ * hw)),
        reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * num_output_ * hw))};
    Dtype* buf_transpose =
        reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * num_output_ * hw));
    kiss_fft_cpx* fftSpace[2] = {
        reinterpret_cast<kiss_fft_cpx*>(malloc(sizeof(kiss_fft_cpx) *
                                               num_complex_out * hw)),
        reinterpret_cast<kiss_fft_cpx*>(malloc(sizeof(kiss_fft_cpx) *
                                               num_complex_out * hw)) };

    for (int b = 0; b < bottom[0]->shape(0); ++b) {
        for (int ipoly = 0; ipoly < 2; ++ipoly) {
            // count and transpose
            // The typical case: 32*8192*28*28, timing = 2.6s
            CPUCountAndTranspose(randh_[ipoly], rands_[ipoly],
                    bottom_data[ipoly] + b * bottom_step[ipoly],
                    batchSpace[ipoly], hw, buf_transpose);
            // fft
            // timing = 6.2s
            caffe_cpu_fft_batch(fft_cfg_noinv, batchSpace[ipoly],
                    fftSpace[ipoly], num_output_, buf_float, hw);
        }
        // multiply
        // timing = 0.3s
        fft_cpx_mul<Dtype>(num_complex_out * hw, fftSpace[0], fftSpace[1],
                fftSpace[0]);
        // ifft
        // timing = 2.8s
        caffe_cpu_ifft_batch(fft_cfg_inverse, batchSpace[0], fftSpace[0],
                num_output_, buf_float, hw);
        // transpose back
        Dtype* out_target;
        if (sum_pool_)
            out_target = batchSpace[1];
        else
            out_target = top_data + b * top_step;

        // transpose back to shape num_output_*hw
        caffe_cpu_transpose(hw, num_output_, batchSpace[0], out_target);

        if (sum_pool_) {
            // sum the columns
            Dtype* topa = top_data + b * top_step;
            for (int i = 0; i < num_output_; ++i) {
                topa[i] = 0.0;
                for (int loc = 0; loc < hw; ++loc)
                    topa[i] += out_target[i * hw + loc];
            }
        }
    }

    free(buf_transpose);
    for (int i = 0; i < 2; ++i) {
        free(batchSpace[i]);
        free(fftSpace[i]);
    }
}

template<typename Dtype>
void CompactBilinearLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
    CHECK_EQ(num_output_ % 2 , 0) <<
            "CPU version don't support odd num_output_";

    if ((!propagate_down[0]) && (!propagate_down[1]))
        return;
    // process the same bottom case
    // when the two bottoms are the same, one propagate down requires the other
    vector<bool> pd = propagate_down;
    if (bottom[0] == bottom[1])
        pd[0] = pd[1] = true;

    const Dtype* bottom_data[2] =
            { bottom[0]->cpu_data(), bottom[1]->cpu_data() };
    // in case two bottom are the same, we should make
    // our algorithm add to bottom_diff
    Dtype* bottom_diff[2] = { bottom[0]->mutable_cpu_diff(), bottom[1]
            ->mutable_cpu_diff() };
    for (int i = 0; i < 2; ++i)
        caffe_set(bottom[i]->count(), Dtype(0.0), bottom_diff[i]);
    const Dtype* top_diff = top[0]->cpu_diff();

    // steps
    const int bottom_step[2] = { bottom[0]->count(1), bottom[1]->count(1) };
    const int top_step = top[0]->count(1);

    const int C[2] = { bottom[0]->shape(1), bottom[1]->shape(1) };
    const int hw = bottom[0]->count(2);

    // the pointer to the (repeated) derivative
    Dtype* dzdy;
    dzdy = reinterpret_cast<Dtype*>(malloc(num_output_ * hw * sizeof(Dtype)));
    // fft[0] for derivative, fft[1] for data
    kiss_fft_cpx* fftSpace[2];
    for (int ipoly = 0; ipoly < 2; ++ipoly)
        fftSpace[ipoly] = reinterpret_cast<kiss_fft_cpx*>(
                malloc(num_complex_out * hw * sizeof(kiss_fft_cpx)));
    Dtype* buf_transpose = reinterpret_cast<Dtype*>(
                malloc(sizeof(Dtype) * num_output_ * hw));

    for (int b = 0; b < bottom[0]->shape(0); ++b) {
        // (copy and) transpose the derivative
        if (sum_pool_) {
            // copy and transpose
            // input: num_output, output: hw*num_output
            for (int ihw = 0; ihw < hw; ++ihw)
                for (int iout = 0; iout < num_output_; ++iout)
                    dzdy[ihw * num_output_ + iout] =
                            top_diff[b * top_step + iout];
        } else {
            caffe_cpu_transpose(num_output_, hw, top_diff + b * top_step, dzdy);
        }

        // fft the derivative, stored in fftSpace[0]
        caffe_cpu_fft_batch(fft_cfg_noinv, dzdy, fftSpace[0], num_output_,
                buf_float, hw);

        for (int ipoly = 0; ipoly < 2; ++ipoly)
            if (pd[1 - ipoly]) {
                // some short hands
                CPUCountAndTranspose(randh_[ipoly], rands_[ipoly],
                        bottom_data[ipoly] + b * bottom_step[ipoly], dzdy, hw,
                        buf_transpose);
                // dzdy's shape is: hw*num_output

                // fliplr(:, 2:end)
                for (int loc = 0; loc < hw; ++loc) {
                    Dtype* pixel = dzdy + loc * num_output_;
                    for (int c = 1; c <= (num_output_ / 2); ++c)
                        std::swap(pixel[c], pixel[num_output_ - c]);
                }

                // fft data
                caffe_cpu_fft_batch(fft_cfg_noinv, dzdy, fftSpace[1],
                        num_output_, buf_float, hw);

                // elementwise mul
                fft_cpx_mul<Dtype>(num_complex_out * hw, fftSpace[0],
                        fftSpace[1], fftSpace[1]);

                // ifft, again reuse dzdy
                caffe_cpu_ifft_batch(fft_cfg_inverse, dzdy, fftSpace[1],
                        num_output_, buf_float, hw);

                // transpose and assign back
                // input: hw*num_input_; output: bottom_diff[1-ipoly] size C*hw
                Dtype* out_diff = bottom_diff[1 - ipoly]
                        + b * bottom_step[1 - ipoly];
                const int* hh = randh_[1 - ipoly].cpu_data();
                const Dtype* ss = rands_[1 - ipoly].cpu_data();

                for (int ic = 0; ic < C[1 - ipoly]; ++ic)
                    for (int loc = 0; loc < hw; ++loc)
                        out_diff[ic * hw + loc] += dzdy[loc * num_output_
                                + hh[ic]] * ss[ic];
            }
    }
    free(dzdy);
    free(fftSpace[0]);
    free(fftSpace[1]);
    free(buf_transpose);
}

#ifdef CPU_ONLY
STUB_GPU(CompactBilinearLayer);
#endif

INSTANTIATE_CLASS(CompactBilinearLayer);
REGISTER_LAYER_CLASS(CompactBilinear);

}  // namespace caffe
