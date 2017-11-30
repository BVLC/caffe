#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lcn_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNLCNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();

  CUDNN_CHECK(cudnnDivisiveNormalizationForward(
      handle_, norm_desc_, CUDNN_DIVNORM_PRECOMPUTED_MEANS,
      cudnn::dataType<Dtype>::one, bottom_desc_, bottom_data,
      NULL, // srcMeansData
      this->tempData1->mutable_gpu_data(), this->tempData2->mutable_gpu_data(),
      cudnn::dataType<Dtype>::zero, top_desc_, top_data));
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNLCNLayer);

} // namespace caffe
#endif
