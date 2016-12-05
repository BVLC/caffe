#include <algorithm>
#include <vector>
#include "caffe/greentea/greentea.hpp"
#ifdef USE_LIBDNN

#include "caffe/layers/libdnn_pool_layer.hpp"

namespace caffe {

template <typename Dtype>
void LibDNNPoolingLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);

  Reshape(bottom, top);
}

template <typename Dtype>
void LibDNNPoolingLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  PoolingLayer<Dtype>::Reshape(bottom, top);

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

    std::vector<int_tp> kernel_vec;
    std::vector<int_tp> pad_vec;
    std::vector<int_tp> stride_vec;
    std::vector<int_tp> dilation_vec;

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
                "cl_khr_int32_base_atomics") ||
            this->device_->CheckCapability(
                "cl_khr_global_int32_base_atomics") ||
            this->device_->CheckCapability(
                "cl_khr_global_int32_extended_atomics"))) ||
        (std::is_same<Dtype, double>::value
        && (this->device_->CheckCapability("cl_khr_int64_base_atomics") ||
            this->device_->CheckCapability("cl_khr_int64_extended_atomics")))) {
      config.bwalgo = LIBDNN_POOLING_BW_ALGO_ATOMIC;
    } else {
      config.bwalgo = LIBDNN_POOLING_BW_ALGO_DIRECT;
    }

    LibDNNPool<Dtype>* libdnn = new LibDNNPool<Dtype>(config);

    libdnn_.reset(libdnn);
  }
}

template <typename Dtype>
LibDNNPoolingLayer<Dtype>::~LibDNNPoolingLayer() {
}

template <typename Dtype>
void LibDNNPoolingLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const bool use_top_mask = top.size() > 1;

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int_tp count = top[0]->count();

  bool test_mode = this->phase_ == caffe::TEST;

  int_tp* mask = nullptr;
  Dtype* top_mask = nullptr;
  Dtype* rand_idx = nullptr;


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
        if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
          caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
            this->rand_idx_.mutable_gpu_data());
#endif  // USE_CUDA
        } else {
#ifdef USE_GREENTEA
          greentea_gpu_rng_uniform(this->device_->id(), count,
            Dtype(0), Dtype(1),
            (cl_mem)(this->rand_idx_.mutable_gpu_data()), 0);
#endif  // USE_GREENTEA
        }
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

template <typename Dtype>
void LibDNNPoolingLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const bool use_top_mask = top.size() > 1;

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int_tp count = bottom[0]->count();

  const int_tp* mask = nullptr;
  const Dtype* top_mask = nullptr;
  const Dtype* rand_idx = nullptr;


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

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_gpu_set(count, Dtype(0.), bottom_diff);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    greentea_gpu_set(this->device_->id(), count, Dtype(0.),
        (cl_mem) bottom_diff, 0);
#endif  // USE_GREENTEA
  }

  libdnn_.get()->Backward(top_diff,
                          bottom_diff,
                          bottom[0]->shape()[1],
                          bottom[0]->shape()[0],
                          mask, top_mask, rand_idx);
}


INSTANTIATE_CLASS(LibDNNPoolingLayer);


}   // namespace caffe
#endif  // USE_LIBDNN
