#include <algorithm>
#include <vector>

#ifdef USE_LIBDNN

#include "caffe/layers/libdnn_pool_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNPoolingLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  PoolingLayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);
  Reshape(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNPoolingLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  PoolingLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);

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

    LibDNNPoolConfig config;
    config.dev_ptr = this->device_;
    config.in_shape = bottom[0]->shape();
    config.out_shape = top[0]->shape();
    config.kernel = kernel_vec;
    config.pad = pad_vec;
    config.stride = stride_vec;
    config.dilation = dilation_vec;
    config.fast_unsafe_math = true;
    config.use_top_mask = (top.size() > 1);

    if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_MAX) {
      config.pool_method = LIBDNN_POOLING_METHOD_MAX;
    }

    if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_AVE) {
      config.pool_method = LIBDNN_POOLING_METHOD_AVE;
    }

    if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_STOCHASTIC) {
      config.pool_method = LIBDNN_POOLING_METHOD_STO;
    }

    config.global_pooling = this->global_pooling_;


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
      config.bwalgo = LIBDNN_POOLING_BW_ALGO_ATOMIC;
    } else {
      config.bwalgo = LIBDNN_POOLING_BW_ALGO_DIRECT;
    }

    LibDNNPool<MItype, MOtype>* libdnn =
        new LibDNNPool<MItype, MOtype>(config);

    libdnn_.reset(libdnn);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
LibDNNPoolingLayer<Dtype, MItype, MOtype>::~LibDNNPoolingLayer() {
}

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNPoolingLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {

  const bool use_top_mask = top.size() > 1;

  vptr<const MItype> bottom_data = bottom[0]->gpu_data();
  vptr<MOtype> top_data = top[0]->mutable_gpu_data();
  int_tp count = top[0]->count();

  bool test_mode = this->phase_ == caffe::TEST;

  vptr<int_tp> mask;
  vptr<MOtype> top_mask;
  vptr<Dtype> rand_idx;

  switch (this->layer_param_.pooling_param().pool()) {
    case PoolingParameter_PoolMethod_MAX:
      if (use_top_mask) {
        top_mask = top[1]->mutable_gpu_data();
      } else {
        mask = this->max_idx_.mutable_gpu_data();
      }
      break;
    case PoolingParameter_PoolMethod_STOCHASTIC:
      if (!test_mode) {
          this->device_->template rng_uniform<Dtype>(count, Dtype(0), Dtype(1),
            this->rand_idx_.mutable_gpu_data());
        rand_idx = this->rand_idx_.mutable_gpu_data();
      }
      break;
  }

  libdnn_.get()->Forward(bottom_data,
                         top_data,
                         bottom[0]->shape()[1],
                         bottom[0]->shape()[0],
                         test_mode,
                         mask, top_mask, rand_idx);
}

template<typename Dtype, typename MItype, typename MOtype>
void LibDNNPoolingLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {

  const bool use_top_mask = top.size() > 1;

  vptr<const MOtype> top_diff = top[0]->gpu_diff();
  vptr<MItype> bottom_diff = bottom[0]->mutable_gpu_diff();
  const int_tp count = bottom[0]->count();

  vptr<const int_tp> mask;
  vptr<const MOtype> top_mask;
  vptr<const Dtype> rand_idx;


  switch (this->layer_param_.pooling_param().pool()) {
    case PoolingParameter_PoolMethod_MAX:
      if (use_top_mask) {
        top_mask = top[1]->gpu_data();
      } else {
        mask = this->max_idx_.gpu_data();
      }
      break;
    case PoolingParameter_PoolMethod_STOCHASTIC:
      rand_idx = this->rand_idx_.gpu_data();
      break;
  }

  this->device_->set(count, (MItype)0, bottom_diff);

  libdnn_.get()->Backward(top_diff,
                          bottom_diff,
                          bottom[0]->shape()[1],
                          bottom[0]->shape()[0],
                          mask, top_mask, rand_idx);
}

INSTANTIATE_CLASS_3T_GUARDED(LibDNNPoolingLayer, (half_fp), (half_fp),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LibDNNPoolingLayer, (float), (float),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LibDNNPoolingLayer, (double), (double),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LibDNNPoolingLayer, (uint8_t), (uint8_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LibDNNPoolingLayer, (uint16_t), (uint16_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LibDNNPoolingLayer, (uint32_t), (uint32_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LibDNNPoolingLayer, (uint64_t), (uint64_t),
                             PROTO_TYPES);

REGISTER_LAYER_CLASS(LibDNNPooling);

}   // namespace caffe
#endif  // USE_LIBDNN
