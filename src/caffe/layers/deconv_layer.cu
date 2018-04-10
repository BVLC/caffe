#include <vector>

#include "caffe/layers/deconv_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void DeconvolutionLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();
  for (int_tp i = 0; i < bottom.size(); ++i) {
    vptr<const Dtype> bottom_data = bottom[i]->gpu_data();
    vptr<Dtype> top_data = top[i]->mutable_gpu_data();
    for (int_tp n = 0; n < this->num_; ++n) {
      this->backward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                              top_data + n * this->top_dim_);
    }
    for (int_tp n = 0; n < this->num_; ++n) {
      if (this->bias_term_) {
        vptr<const Dtype> bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void DeconvolutionLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();
  vptr<Dtype> weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int_tp i = 0; i < top.size(); ++i) {
    vptr<const Dtype> top_diff = top[i]->gpu_diff();
    vptr<const Dtype> bottom_data = bottom[i]->gpu_data();
    vptr<Dtype> bottom_diff = bottom[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      vptr<Dtype> bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int_tp n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(top_diff + n * this->top_dim_, bottom_data +
                                n * this->bottom_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->forward_gpu_gemm(top_diff + n * this->top_dim_, weight,
                                 bottom_diff + n * this->bottom_dim_,
                                 this->param_propagate_down_[0]);
        }
      }
    }
  }
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Forward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Forward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Forward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Forward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Forward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Backward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Backward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Backward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Backward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DeconvolutionLayer, Backward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));
}  // namespace caffe
