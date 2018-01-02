/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef MKLDNN_SUPPORTED
#include "caffe/mkldnn_memory.hpp"

namespace caffe {


shared_ptr<MKLDNNStream> StreamHolder::get_stream()
{
    if (this->_current_stream == NULL || !this->_current_stream->ready()) {
        _current_stream.reset(new MKLDNNStream());
    }
    return _current_stream;
}

template <typename Dtype>
shared_ptr<MKLDNNStream>  MKLDNNPrimitive<Dtype>::get_mkldnn_stream() {
    if(mkldnn_stream == NULL)
        mkldnn_stream = StreamHolder::Instance().get_stream();
    else
        StreamHolder::Instance().prepare_mkldnn_stream(mkldnn_stream);
    return mkldnn_stream;

}

template <typename Dtype>
shared_ptr<MKLDNNStream>  MKLDNNPrimitive<Dtype>::submit() {
    CHECK(this->aprimitive);
    this->get_mkldnn_stream()->submit({*(this->aprimitive)});
    return mkldnn_stream;
}

template <typename Dtype>
MKLDNNLayer<Dtype>::MKLDNNLayer(const LayerParameter &param) :
  BaseQuantLayer<Dtype>() {
  if (param.has_quantization_param()) {
    this->precision_ = param.quantization_param().precision();
    this->rounding_ = param.quantization_param().rounding_scheme();
    switch (this->precision_) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      this->bw_layer_in_ = param.quantization_param().bw_layer_in();
      this->bw_layer_out_ = param.quantization_param().bw_layer_out();
      this->bw_params_ = param.quantization_param().bw_params();
      for (int i = 0; i < param.quantization_param().fl_layer_in_size(); i++)
        this->fl_layer_in_.push_back(param.quantization_param().fl_layer_in(i));
      for (int i = 0; i < param.quantization_param().fl_layer_out_size(); i++)
        this->fl_layer_out_.push_back(param.quantization_param().fl_layer_out(i));
      for (int i = 0; i < param.quantization_param().fl_params_size(); i++)
        this->fl_params_.push_back(param.quantization_param().fl_params(i));
      //floating point
      for (int i = 0; i < param.quantization_param().scale_in_size(); i++)
        this->scale_in_.push_back(param.quantization_param().scale_in(i));
      for (int i = 0; i < param.quantization_param().scale_out_size(); i++)
        this->scale_out_.push_back(param.quantization_param().scale_out(i));
      for (int i = 0; i < param.quantization_param().scale_params_size(); i++)
        this->scale_params_.push_back(param.quantization_param().scale_params(i));

      break;
    default:
      LOG(FATAL) << "Unknown precision mode: " << this->precision_;
      break;
    }
  }
}

template class MKLDNNLayer<double>;
template class MKLDNNLayer<float>;
template class MKLDNNPrimitive<double>;
template class MKLDNNPrimitive<float>;
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
