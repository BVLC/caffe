#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lcn_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNLCNLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data().get_cuda_ptr();
  Dtype* top_data = top[0]->mutable_gpu_data().get_cuda_ptr();

  CUDNN_CHECK(cudnnDivisiveNormalizationForward(
        handle_, norm_desc_, CUDNN_DIVNORM_PRECOMPUTED_MEANS,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        NULL,  // srcMeansData
        this->tempData1, this->tempData2,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data) );
}

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNLCNLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff().get_cuda_ptr();
  const Dtype* top_data = top[0]->gpu_data().get_cuda_ptr();
  const Dtype* bottom_data = bottom[0]->gpu_data().get_cuda_ptr();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff().get_cuda_ptr();

  CUDNN_CHECK(cudnnDivisiveNormalizationBackward(
        handle_, norm_desc_, CUDNN_DIVNORM_PRECOMPUTED_MEANS,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        NULL, top_diff,  // NULL - srcMeansData
        this->tempData1, this->tempData2,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_diff,
        NULL) );
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNLCNLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNLCNLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNLCNLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CuDNNLCNLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
#endif
