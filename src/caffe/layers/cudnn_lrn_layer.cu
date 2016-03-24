#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lrn_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNLRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  CUDNN_CHECK(cudnnLRNCrossChannelForward(
        Caffe::cudnn_handle(), norm_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data) );
}

template <typename Dtype>
void CuDNNLRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  CUDNN_CHECK(cudnnLRNCrossChannelBackward(
        Caffe::cudnn_handle(), norm_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        cudnn::dataType<Dtype>::one,
        top_desc_, top_data,
        top_desc_, top_diff,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_diff) );
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNLRNLayer);

};  // namespace caffe

#endif
