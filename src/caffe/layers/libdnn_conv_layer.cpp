#include <algorithm>
#include <vector>

#ifdef USE_LIBDNN

#include "caffe/layers/libdnn_conv_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNConvolutionLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  BaseConvolutionLayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);
  this->use_colbuffer_ = false;
  Reshape(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNConvolutionLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  this->use_colbuffer_ = false;
  BaseConvolutionLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);

  bool shapes_changed = false;
  if (libdnn_.get() != nullptr) {
    vector<int_tp> libdnn_in_sh = libdnn_.get()->get_config().in_shape;
    vector<int_tp> libdnn_out_sh = libdnn_.get()->get_config().out_shape;
    const vector<int_tp>& new_in_sh = bottom[0]->shape();
    const vector<int_tp>& new_out_sh = top[0]->shape();
    bool in_eq = libdnn_in_sh.size() == new_in_sh.size()
                 && libdnn_in_sh[0] >= new_in_sh[0] 
                 && std::equal(libdnn_in_sh.begin() + 1,
                               libdnn_in_sh.end(), new_in_sh.begin() + 1);
    bool out_eq = libdnn_out_sh.size() == new_out_sh.size()
                 && libdnn_out_sh[0] >= new_out_sh[0] 
                 && std::equal(libdnn_out_sh.begin() + 1,
                               libdnn_out_sh.end(),new_out_sh.begin() + 1);
    shapes_changed = !in_eq || !out_eq;
  }

  if (libdnn_.get() == nullptr || shapes_changed) {
    int_tp* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
    int_tp* pad_data = this->pad_.mutable_cpu_data();
    int_tp* stride_data = this->stride_.mutable_cpu_data();
    int_tp* dilation_data = this->dilation_.mutable_cpu_data();

    vector<int_tp> kernel_vec;
    vector<int_tp> pad_vec;
    vector<int_tp> stride_vec;
    vector<int_tp> dilation_vec;

    for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
        kernel_vec.push_back(kernel_shape_data[i]);
        pad_vec.push_back(pad_data[i]);
        stride_vec.push_back(stride_data[i]);
        dilation_vec.push_back(dilation_data[i]);
    }

    LibDNNConvConfig config;
    config.dev_ptr = this->device_;
    config.in_shape = bottom[0]->shape();
    config.out_shape = top[0]->shape();
    config.kernel = kernel_vec;
    config.pad = pad_vec;
    config.stride = stride_vec;
    config.dilation = dilation_vec;
    config.group = this->group_;
    config.bias_term = this->bias_term_;
    config.fast_unsafe_math = true;
    config.weights_backward = this->param_propagate_down_[0];
    config.bias_backward = this->param_propagate_down_[1];

    if ((std::is_same<Dtype, float>::value
        && (this->device_->CheckCapability(
              DEVICE_INT64_GLOBAL_ATOMICS_SUPPORT) ||
            this->device_->CheckCapability(
             DEVICE_INT64_GLOBAL_EXTENDED_ATOMICS_SUPPORT))) ||
        (std::is_same<Dtype, double>::value
        && (this->device_->CheckCapability(
            DEVICE_INT64_GLOBAL_ATOMICS_SUPPORT) ||
            this->device_->CheckCapability(
                DEVICE_INT64_GLOBAL_EXTENDED_ATOMICS_SUPPORT)))) {
      config.wgalgo = LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC;
      config.bwalgo = LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC;
    } else {
      config.wgalgo = LIBDNN_CONVOLUTION_WG_ALGO_DIRECT;
      config.bwalgo = LIBDNN_CONVOLUTION_BW_ALGO_IM2COL;
    }

    LibDNNConv<MItype, MOtype>* libdnn =
        new LibDNNConv<MItype, MOtype>(config);

    libdnn_.reset(libdnn);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
LibDNNConvolutionLayer<Dtype, MItype, MOtype>::~LibDNNConvolutionLayer() {
}

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNConvolutionLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {

  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();
  vptr<const Dtype> bias;
  Dtype bias_mult;
  if (this->bias_term_) {
     bias_mult = this->bias_multiplier_.cpu_data()[0];
     bias = this->blobs_[1]->gpu_data();
  }

  for (int_tp i = 0; i < bottom.size(); ++i) {
    const QuantizerValues* const bottom_quant =
        &(this->bottom_quants_[i]->out_quantizer_values());
    const QuantizerValues* const weight_quant =
        this->blobs_quants_.size() > 0 ?
            &(this->blobs_quants_[0]->out_quantizer_values()) : nullptr;
    const QuantizerValues* const bias_quant =
        this->blobs_quants_.size() > 1 ?
            &(this->blobs_quants_[1]->out_quantizer_values()) : nullptr;
    const QuantizerValues* const top_quant =
        &(this->top_quants_[i]->in_quantizer_values());

    vptr<const MItype> bottom_data = bottom[i]->gpu_data();
    vptr<MOtype> top_data = top[i]->mutable_gpu_data();
    libdnn_.get()->Forward(bottom_data, weight, bias_mult,
                           bias, top_data, bottom[i]->shape()[0],
                           bottom_quant, weight_quant,
                           &(this->bias_multiplier_qv_),
                           bias_quant, top_quant);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNConvolutionLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {

  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();
  vptr<const Dtype> bias;
  vptr<Dtype> weight_diff = this->blobs_[0]->mutable_gpu_diff();
  vptr<Dtype> bias_diff;
  Dtype bias_mult;
  if (this->bias_term_) {
     bias = this->blobs_[1]->gpu_data();
     bias_diff = this->blobs_[1]->mutable_gpu_diff();
     bias_mult = this->bias_multiplier_.cpu_data()[0];
  }

  for (int_tp i = 0; i < top.size(); ++i) {
    vptr<const Dtype> top_data = top[i]->gpu_data();
    vptr<const Dtype> top_diff = top[i]->gpu_diff();
    vptr<const Dtype> bottom_data = bottom[i]->gpu_data();
    vptr<Dtype> bottom_diff = bottom[i]->mutable_gpu_diff();
    libdnn_.get()->Backward(propagate_down[i], propagate_down[i] ||
                            (this->param_propagate_down_[0] ||
                             this->param_propagate_down_[1]),
                            top_data, top_diff,
                            weight, weight_diff,
                            bias_mult, bias, bias_diff,
                            bottom_data, bottom_diff,
                            bottom[i]->shape()[0]);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNConvolutionLayer<Dtype, MItype, MOtype>::Tune(
          vptr<Dtype> top_data, vptr<Dtype> top_diff,
          vptr<Dtype> bottom_data, vptr<Dtype> bottom_diff,
          int_tp batch_size) {
  vptr<Dtype> weight_data = this->blobs_[0]->mutable_gpu_data();
  vptr<Dtype> weight_diff = this->blobs_[0]->mutable_gpu_diff();
  vptr<Dtype> bias_data;
  vptr<Dtype> bias_diff;
  Dtype bias_mult;
  if (this->bias_term_) {
     bias_data = this->blobs_[1]->mutable_gpu_data();
     bias_diff = this->blobs_[1]->mutable_gpu_diff();
     bias_mult = this->bias_multiplier_.cpu_data()[0];
  }

  libdnn_.get()->Tune(top_data, top_diff,
                      weight_data, weight_diff,
                      bias_mult, bias_data, bias_diff,
                      bottom_data, bottom_diff,
                      batch_size);
}

INSTANTIATE_CLASS_3T_GUARDED(LibDNNConvolutionLayer, (half_fp), (half_fp),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LibDNNConvolutionLayer, (float), (float),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LibDNNConvolutionLayer, (double), (double),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LibDNNConvolutionLayer, (uint8_t), (uint8_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LibDNNConvolutionLayer, (uint16_t), (uint16_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LibDNNConvolutionLayer, (uint32_t), (uint32_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LibDNNConvolutionLayer, (uint64_t), (uint64_t),
                             PROTO_TYPES);

REGISTER_LAYER_CLASS(LibDNNConvolution);


}   // namespace caffe
#endif  // USE_LIBDNN
