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

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/signed_sqrt_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void caffe_gpu_signed_sqrt(const int nthreads,
        const Dtype* src, Dtype* dst) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        if (src[index] >= 0)
            dst[index] = sqrt(src[index]);
        else
            dst[index] = -sqrt(-src[index]);
    }
}


template <typename Dtype>
void SignedSqrtLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

  caffe_gpu_signed_sqrt<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
  <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, top_data);
}

template <typename Dtype>
void SignedSqrtLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    if (propagate_down[0]) {
        const int count = bottom[0]->count();

        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* top_data = top[0]->gpu_data();

        caffe_gpu_abs(count, top_data, bottom_diff);
        caffe_gpu_add_scalar(count, epsilon, bottom_diff);
        caffe_gpu_div(count, top_diff, bottom_diff, bottom_diff);
        caffe_gpu_scal(count, Dtype(0.5), bottom_diff);
      }
}

INSTANTIATE_LAYER_GPU_FUNCS(SignedSqrtLayer);


}  // namespace caffe
