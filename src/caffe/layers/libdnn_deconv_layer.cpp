#include <algorithm>
#include <vector>

#ifdef USE_LIBDNN

#include "caffe/layers/libdnn_deconv_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNDeconvolutionLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  DeconvolutionLayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);
  this->use_colbuffer_ = false;
  Reshape(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNDeconvolutionLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  this->use_colbuffer_ = false;
  DeconvolutionLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);

  bool shapes_changed = false;
  if (libdnn_.get() != nullptr) {
    shapes_changed = shapes_changed || (libdnn_.get()->get_config().in_shape
        != bottom[0]->shape());
    shapes_changed = shapes_changed || (libdnn_.get()->get_config().out_shape
        != top[0]->shape());
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

    LibDNNDeconvConfig config;
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

    // Atomic algorithm requirements:
    // - Float & 32 bit atomics available
    // - Double & 64 bit atomics available
    // - No bias term
    if (((std::is_same<Dtype, float>::value
        && (this->device_->CheckCapability(
                "cl_khr_int32_base_atomics") ||
            this->device_->CheckCapability(
                "cl_khr_global_int32_base_atomics") ||
            this->device_->CheckCapability(
                "cl_khr_global_int32_extended_atomics"))) ||
        (std::is_same<Dtype, double>::value
        && (this->device_->CheckCapability("cl_khr_int64_base_atomics") ||
            this->device_->CheckCapability("cl_khr_int64_extended_atomics"))))
            && !this->bias_term_) {
      config.wgalgo = LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC;
      config.bwalgo = LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC;
    } else {
      config.wgalgo = LIBDNN_CONVOLUTION_WG_ALGO_DIRECT;
      config.bwalgo = LIBDNN_CONVOLUTION_BW_ALGO_IM2COL;
    }

    LibDNNDeconv<Dtype, MItype, MOtype>* libdnn =
        new LibDNNDeconv<Dtype, MItype, MOtype>(config);

    libdnn_.reset(libdnn);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
LibDNNDeconvolutionLayer<Dtype, MItype, MOtype>::~LibDNNDeconvolutionLayer() {
}

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNDeconvolutionLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {

  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();
  vptr<const Dtype> bias;
  if (this->bias_term_) {
     bias = this->blobs_[1]->gpu_data();
  }

  for (int_tp i = 0; i < bottom.size(); ++i) {
    vptr<const MItype> bottom_data = bottom[i]->gpu_data();
    vptr<MOtype> top_data = top[i]->mutable_gpu_data();
    libdnn_.get()->Forward(bottom_data, weight, bias,
                           top_data, bottom[i]->shape()[0]);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNDeconvolutionLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {

  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();
  vptr<const Dtype> bias;
  vptr<Dtype> weight_diff = this->blobs_[0]->mutable_gpu_diff();
  vptr<Dtype> bias_diff;
  if (this->bias_term_) {
     bias = this->blobs_[1]->gpu_data();
     bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }

  for (int_tp i = 0; i < top.size(); ++i) {
    vptr<const MOtype> top_data = top[i]->gpu_data();
    vptr<const MOtype> top_diff = top[i]->gpu_diff();
    vptr<const MItype> bottom_data = bottom[i]->gpu_data();
    vptr<MItype> bottom_diff = bottom[i]->mutable_gpu_diff();
    libdnn_.get()->Backward(propagate_down[i], propagate_down[i] ||
                            (this->param_propagate_down_[0] ||
                             this->param_propagate_down_[1]),
                            top_data, top_diff,
                            weight, weight_diff,
                            bias, bias_diff,
                            bottom_data, bottom_diff,
                            bottom[i]->shape()[0]);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNDeconvolutionLayer<Dtype, MItype, MOtype>::Tune(
          vptr<MOtype> top_data, vptr<MOtype> top_diff,
          vptr<MItype> bottom_data, vptr<MItype> bottom_diff,
          int_tp batch_size) {
  vptr<Dtype> weight_data = this->blobs_[0]->mutable_gpu_data();
  vptr<Dtype> weight_diff = this->blobs_[0]->mutable_gpu_diff();
  vptr<Dtype> bias_data;
  vptr<Dtype> bias_diff;
  if (this->bias_term_) {
     bias_data = this->blobs_[1]->mutable_gpu_data();
     bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }

  libdnn_.get()->Tune(top_data, top_diff,
                      weight_data, weight_diff,
                      bias_data, bias_diff,
                      bottom_data, bottom_diff,
                      batch_size);
}


INSTANTIATE_CLASS_3T_GUARDED(LibDNNDeconvolutionLayer,
                             (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(LibDNNDeconvolutionLayer,
                             (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(LibDNNDeconvolutionLayer,
                             (double), (double), (double));

}   // namespace caffe

#endif  // USE_LIBDNN
