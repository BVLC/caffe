#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ConvolutionLayer<Dtype, MItype, MOtype>::Forward_gpu(
      const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();
  for (int_tp i = 0; i < bottom.size(); ++i) {
    vptr<const Dtype> bottom_data = bottom[i]->gpu_data();
    vptr<Dtype> top_data = top[i]->mutable_gpu_data();
    // Multi queue execution, all previous work needs to be done first
    this->device_->FinishQueues();
    for (int_tp n = 0; n < this->num_; ++n) {
      // Multi queue execution, go through work queues
      this->device_->SwitchQueue(n);
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                             top_data + n * this->top_dim_, false,
                             &(this->bottom_quants_[i]->out_quantizer_values()),
                             &(this->blobs_quants_[0]->out_quantizer_values()),
                             &(this->top_quants_[i]->in_quantizer_values()));
      if (this->bias_term_) {
        vptr<const Dtype> bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias,
                             &(this->top_quants_[i]->in_quantizer_values()),
                             &(this->blobs_quants_[1]->out_quantizer_values()));
      }
    }
    // Multi queue execution, finish all queues
    this->device_->FinishQueues();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void ConvolutionLayer<Dtype, MItype, MOtype>::Backward_gpu(
      const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom) {
  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();
  vptr<Dtype> weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int_tp i = 0; i < top.size(); ++i) {
    vptr<const Dtype> top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      vptr<Dtype> bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      vptr<const Dtype> bottom_data = bottom[i]->gpu_data();
      vptr<Dtype> bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
                                top_diff + n * this->top_dim_, weight_diff);
        }
      }
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[i]) {
        // Multi queue execution, all previous work needs to be done first
        this->device_->FinishQueues();
        for (int_tp n = 0; n < this->num_; ++n) {
          // Multi queue execution, go through work queues
          this->device_->SwitchQueue(n);
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
                                  bottom_diff + n * this->bottom_dim_);
        }
        // Multi queue execution, finish all queues
        this->device_->FinishQueues();
      }
    }
  }
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Forward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Forward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Forward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Forward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Forward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Backward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Backward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Backward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Backward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConvolutionLayer, Backward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));
}  // namespace caffe
