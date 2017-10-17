/*
Copyright (c) 2016. The Regents of the University of California (Regents). All
Rights Reserved. Permission to use, copy, modify, and distribute this software
and its documentation for educational, research, not-for-profit, and commercial
purposes (such rights not subject to transfer), without fee, and without a
signed licensing agreement, is hereby granted, provided that the above copyright
notice, this paragraph and the following two paragraphs appear in all copies,
modifications, and distributions. Contact The Office of Technology Licensing,
UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
(510) 643-7201, for commercial licensing opportunities.

Yang Gao, University of California, Berkeley.


IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS
PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#ifndef CAFFE_COMPACT_BILINEAR_LAYER_HPP_
#define CAFFE_COMPACT_BILINEAR_LAYER_HPP_

#ifndef CPU_ONLY
#include <cufft.h>
#endif

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/_kiss_fft_guts.h"
#include "caffe/util/kiss_fftr.h"


namespace caffe {
template<typename Dtype>
struct CaffeComplex {
    Dtype x;
    Dtype y;
};

/**
 * @brief Computes @f$ y = compact_bilinear(x_1, x_2) @f$
 * By default, we do spatial sum pooling. One could specify
 * in parameter, such that it doesn't pool spatially.
 * This implementation don't support learning the random
 * weights. Since in practice, it doesn't improve performance.
 * The weights were generated using a fixed random seed.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C_1 \times H \times W) @f$
 *      the inputs @f$ x_1 @f$
 *   -# @f$ (N \times C_2 \times H \times W) @f$
 *      the inputs @f$ x_2 @f$
 *      the two inputs could be the same blob
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times num_output \times 1(H) \times 1(W)) @f$
 *      the compact bilinear pooled vector @f$ y @f$
 *      the output dimension depends on whether to have spatial
 *      sum pooling (sum_pool).
 *      num_output is the parameter required to be specified.
 */
template<typename Dtype>
class CompactBilinearLayer: public Layer<Dtype> {
 public:
    CompactBilinearLayer(const LayerParameter& param);
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual ~CompactBilinearLayer();

    virtual inline const char* type() const {
        return "CompactBilinear";
    }
    virtual inline int ExactNumBottomBlobs() const {
        return 2;
    }
    virtual inline int ExactNumTopBlobs() const {
        return 1;
    }

 protected:
    /// @copydoc CompactBilinearLayer
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom);

    // parameters cache
    int num_output_;
    bool sum_pool_;

    // internal data
    // random but fixed weights
    Blob<int> randh_[2];
    Blob<Dtype> rands_[2];

    // CPU kiss_fft execution plans
    kiss_fftr_cfg fft_cfg_noinv;
    kiss_fftr_cfg fft_cfg_inverse;
    float* buf_float;
    // the length of fft after the R2C transform
    // ==floor(1.0*num_output_/2)+1
    int num_complex_out;

#ifndef CPU_ONLY
    // whether the GPU specific caches have been initialized.
    bool plan_init;
    // the internal GPU batch size to do the compact bilinear transform
    int batchsz;

    // cufft plans
    cufftHandle plan_noinv_batch;
    cufftHandle plan_noinv_1;
    cufftHandle plan_inv_batch;
    cufftHandle plan_inv_1;
    // all one constant vector for spatial sum pooling
    Dtype* ones_hw;
#endif

    // some internal helpers
 private:
    // CPU helpers
    void CPUCountAndTranspose(const Blob<int>& randh, const Blob<Dtype>& rands,
            const Dtype* bottom, Dtype* top, const int hw, Dtype* temptop);

    // GPU helpers
    void caffe_gpu_fft(const int batchlen, const int hw, const int nfft,
            const Dtype* src, CaffeComplex<Dtype>* output);
    void caffe_gpu_ifft(const int batchlen, const int hw, const int nfft,
            const CaffeComplex<Dtype>* src, Dtype* output);
    void Initializations(const int hw);
};

}  // namespace caffe

#endif  // CAFFE_COMPACT_BILINEAR_LAYER_HPP_
