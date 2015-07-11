#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template<typename Dtype>
void PoolingNDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  // Set the max number of top blobs before calling base Layer::SetUp.
  // If doing MAX pooling, we can optionally output an extra top Blob
  // for the mask.  Otherwise, we only have one top Blob.
  if (this->layer_param_.pooling_param().pool()
      == PoolingParameter_PoolMethod_MAX) {
    max_top_blobs_ = 2;
  } else {
    max_top_blobs_ = 1;
  }
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  CHECK(!(pool_param.kernel_size_size() > 0) !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK((pool_param.kernel_size_size() > 0) ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!(pool_param.pad_size() > 0) && pool_param.has_pad_h()
          && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!(pool_param.stride_size() > 0) && pool_param.has_stride_h()
          && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (pool_param.kernel_size_size() > 0) {
    kernel_h_ = kernel_w_ = pool_param.kernel_size(0);
  } else {
    kernel_h_ = pool_param.kernel_h();
    kernel_w_ = pool_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0)<< "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0)<< "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad_size() > 0 ?
        pool_param.pad(0) : 0;
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  CHECK_EQ(pad_h_, 0);
  CHECK_EQ(pad_w_, 0);
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride_size() > 0 ?
        pool_param.stride(0) : 1;
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
  if (!pool_param.has_kstride_h()) {
    kstride_h_ = kstride_w_ = pool_param.kstride_size() > 0 ?
        pool_param.kstride(0) : 1;
  } else {
    kstride_h_ = pool_param.kstride_h();
    kstride_w_ = pool_param.kstride_w();
  }

  int ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  int ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;

  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  pooled_height_ = static_cast<int>(ceil(
      static_cast<float>(height_ + 2 * pad_h_ - ext_kernel_h) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(
      static_cast<float>(width_ + 2 * pad_w_ - ext_kernel_w) / stride_w_)) + 1;

  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool()
      == PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
                     pooled_width_);
  }
}

template<typename Dtype>
void PoolingNDLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  LayerSetUp(bottom, top);
}

template<typename Dtype>
void PoolingNDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  LOG(FATAL)<< "Forward_cpu() not implemented in PoolingNDLayer.";
}

template<typename Dtype>
void PoolingNDLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL)<< "Backward_cpu() not implemented in PoolingNDLayer.";
  return;
}

#ifdef CPU_ONLY
STUB_GPU(PoolingNDLayer);
#endif

INSTANTIATE_CLASS(PoolingNDLayer);
REGISTER_LAYER_CLASS(PoolingND);

}    // namespace caffe
