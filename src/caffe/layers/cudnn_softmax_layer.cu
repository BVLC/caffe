#ifdef USE_CUDNN
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cudnn_softmax_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNSoftmaxLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data().get_cuda_ptr();
  Dtype* top_data = top[0]->mutable_gpu_data().get_cuda_ptr();
  CUDNN_CHECK(cudnnSoftmaxForward(handle_, CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data));
}

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNSoftmaxLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data().get_cuda_ptr();
    const Dtype* top_diff = top[0]->gpu_diff().get_cuda_ptr();
    const Dtype* bottom_data = bottom[0]->gpu_data().get_cuda_ptr();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff().get_cuda_ptr();

    CUDNN_CHECK(cudnnSoftmaxBackward(handle_, CUDNN_SOFTMAX_ACCURATE,
          CUDNN_SOFTMAX_MODE_CHANNEL,
          cudnn::dataType<Dtype>::one,
          top_desc_, top_data, top_desc_, top_diff,
          cudnn::dataType<Dtype>::zero,
          bottom_desc_, bottom_diff));
  }
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNSoftmaxLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNSoftmaxLayer, Forward_gpu,
                                  (double), (double), (double));


INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNSoftmaxLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNSoftmaxLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
#endif
