#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_relu_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNReLULayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  // Fallback to standard Caffe for leaky ReLU.
  if (ReLULayer<Dtype, MItype, MOtype>::
      layer_param_.relu_param().negative_slope() != 0) {
    return ReLULayer<Dtype, MItype, MOtype>::Forward_gpu(bottom, top);
  }

  const Dtype* bottom_data = bottom[0]->gpu_data().get_cuda_ptr();
  Dtype* top_data = top[0]->mutable_gpu_data().get_cuda_ptr();
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnActivationForward(this->handle_,
        activ_desc_,
        cudnn::dataType<Dtype>::one,
        this->bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        this->top_desc_, top_data));
#else
  CUDNN_CHECK(cudnnActivationForward_v4(this->handle_,
        activ_desc_,
        cudnn::dataType<Dtype>::one,
        this->bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        this->top_desc_, top_data));
#endif
}

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNReLULayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  // Fallback to standard Caffe for leaky ReLU.
  if (ReLULayer<Dtype, MItype, MOtype>::
      layer_param_.relu_param().negative_slope() != 0) {
    return ReLULayer<Dtype, MItype, MOtype>::
        Backward_gpu(top, propagate_down, bottom);
  }

  const Dtype* top_data = top[0]->gpu_data().get_cuda_ptr();
  const Dtype* top_diff = top[0]->gpu_diff().get_cuda_ptr();
  const Dtype* bottom_data = bottom[0]->gpu_data().get_cuda_ptr();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff().get_cuda_ptr();
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnActivationBackward(this->handle_, activ_desc_,
        cudnn::dataType<Dtype>::one,
        this->top_desc_, top_data, this->top_desc_, top_diff,
        this->bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        this->bottom_desc_, bottom_diff));
#else
  CUDNN_CHECK(cudnnActivationBackward_v4(this->handle_,
        activ_desc_,
        cudnn::dataType<Dtype>::one,
        this->top_desc_, top_data, this->top_desc_, top_diff,
        this->bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        this->bottom_desc_, bottom_diff));
#endif
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNReLULayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNReLULayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNReLULayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNReLULayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
#endif
