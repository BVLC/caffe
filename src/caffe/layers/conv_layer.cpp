#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ConvolutionLayer<Dtype, MItype, MOtype>::Forward_cpu(
                                          const vector<Blob<MItype>*>& bottom,
                                          const vector<Blob<MOtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int_tp i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int_tp n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                             top_data + n * this->top_dim_, false,
                             &(this->bottom_quants_[i]->out_quantizer_values()),
                             &(this->blobs_quants_[0]->out_quantizer_values()),
                             &(this->top_quants_[i]->in_quantizer_values()));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias,
                             &(this->top_quants_[i]->in_quantizer_values()),
                             &(this->blobs_quants_[1]->out_quantizer_values()));
      }
    }
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void ConvolutionLayer<Dtype, MItype, MOtype>::Backward_cpu(
                                         const vector<Blob<MOtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<MItype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int_tp i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int_tp n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                                  bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(ConvolutionLayer,
                             (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(ConvolutionLayer,
                             (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(ConvolutionLayer,
                             (double), (double), (double));
INSTANTIATE_CLASS_3T_GUARDED(ConvolutionLayer,
                             (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASS_3T_GUARDED(ConvolutionLayer,
                             (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASS_3T_GUARDED(ConvolutionLayer,
                             (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASS_3T_GUARDED(ConvolutionLayer,
                             (uint64_t), (uint64_t), (uint64_t));

}  // namespace caffe
