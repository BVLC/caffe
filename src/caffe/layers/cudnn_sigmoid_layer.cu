#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_sigmoid_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNSigmoidLayer<Dtype, MItype, MOtype>::Forward_gpu(const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
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
void CuDNNSigmoidLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* top_data = top[0]->gpu_data().get_cuda_ptr();
  const Dtype* top_diff = top[0]->gpu_diff().get_cuda_ptr();
  const Dtype* bottom_data = bottom[0]->gpu_data().get_cuda_ptr();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff().get_cuda_ptr();
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnActivationBackward(this->handle_,
        activ_desc_,
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


INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNSigmoidLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNSigmoidLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNSigmoidLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNSigmoidLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
#endif
