#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // Fallback to Caffe for padded pooling, max top mask.
  if ((this->pad_h_ > 0 || this->pad_w_ > 0) || (*top).size() > 1) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to standard Caffe for padded pooling.";
    return PoolingLayer<Dtype>::Forward_gpu(bottom, top);
  }

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  CUDNN_CHECK(cudnnPoolingForward(handle_, pooling_desc_,
      bottom_desc_, bottom_data, top_desc_, top_data));
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }

  // Fallback to Caffe for padded pooling, max top mask.
  if ((this->pad_h_ > 0 || this->pad_w_ > 0) || top.size() > 1) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to standard Caffe for padded pooling.";
    return PoolingLayer<Dtype>::Backward_gpu(top, propagate_down, bottom);
  }

  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  CUDNN_CHECK(cudnnPoolingBackward(handle_, pooling_desc_,
      top_desc_, top_data, top_desc_, top_diff,
      bottom_desc_, bottom_data, bottom_desc_, bottom_diff));
}

INSTANTIATE_CLASS(CuDNNPoolingLayer);

}  // namespace caffe
#endif
