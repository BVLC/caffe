#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lcn_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNLCNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  temp1_.reserve(tempDataSize_);
  temp2_.reserve(tempDataSize_);

  CUDNN_CHECK(cudnnDivisiveNormalizationForward(
        Caffe::cudnn_handle(), norm_desc_, CUDNN_DIVNORM_PRECOMPUTED_MEANS,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        NULL,  // srcMeansData
        temp1_.data(), temp2_.data(),
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data) );

  temp1_.release();
  temp2_.release();
}

template <typename Dtype>
void CuDNNLCNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  temp1_.reserve(tempDataSize_);
  temp2_.reserve(tempDataSize_);

  CUDNN_CHECK(cudnnDivisiveNormalizationBackward(
        Caffe::cudnn_handle(), norm_desc_,
        CUDNN_DIVNORM_PRECOMPUTED_MEANS,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        NULL, top_diff,  // NULL - srcMeansData
        temp1_.data(), temp2_.data(),
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_diff,
        NULL) );

  temp1_.release();
  temp2_.release();
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNLCNLayer);

}  // namespace caffe
#endif
