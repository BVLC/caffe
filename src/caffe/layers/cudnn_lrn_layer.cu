#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lrn_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNLRNLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data().get_cuda_ptr();
  Dtype* top_data = top[0]->mutable_gpu_data().get_cuda_ptr();

  CUDNN_CHECK(cudnnLRNCrossChannelForward(
        handle_, norm_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data) );
}

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNLRNLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff().get_cuda_ptr();
  const Dtype* top_data = top[0]->gpu_data().get_cuda_ptr();
  const Dtype* bottom_data = bottom[0]->gpu_data().get_cuda_ptr();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff().get_cuda_ptr();

  CUDNN_CHECK(cudnnLRNCrossChannelBackward(
        handle_, norm_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        cudnn::dataType<Dtype>::one,
        top_desc_, top_data,
        top_desc_, top_diff,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_diff) );
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNLRNLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNLRNLayer, Forward_gpu,
                                  (double), (double), (double));


INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNLRNLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNLRNLayer, Backward_gpu,
                                  (double), (double), (double));

};  // namespace caffe

#endif
