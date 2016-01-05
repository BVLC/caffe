#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
DnnLRNLayer<Dtype>::~DnnLRNLayer()
{
  dnnDelete<Dtype>(lrnFwd);
  dnnDelete<Dtype>(lrnBwd);
  dnnReleaseBuffer<Dtype>(lrn_buffer_);
}


template <typename Dtype>
void DnnLRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  size_ = this->layer_param_.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";

  alpha_ = this->layer_param_.lrn_param().alpha();
  beta_ = this->layer_param_.lrn_param().beta();
  k_ = this->layer_param_.lrn_param().k();

  size_t dim = 4, sizes[4], strides[4];

  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();
  num_      = bottom[0]->num();

  sizes[0] = width_;
  sizes[1] = height_;
  sizes[2] = channels_;
  sizes[3] = num_;

  strides[0] = 1;
  strides[1] = sizes[0];
  strides[2] = sizes[0]*sizes[1];
  strides[3] = sizes[0]*sizes[1]*sizes[2];

  dnnError_t e;
  dnnLayout_t lrn_layout   = NULL;
  dnnLayout_t lrn_buffer_l = NULL;
  lrn_buffer_ = NULL;

  if ( size_ == 5 && /*alpha_ == 0.0001 && beta_ == 0.75 &&*/
      ( (channels_ == 96  &&  height_ == 55 && width_ == 55) ||
        (channels_ == 256 &&  height_ == 27 && width_ == 27)
       )
      )
  {
     e = dnnLayoutPCLCreate<Dtype>(&lrn_layout, dim, sizes);
  } else {
     e = dnnLayoutCreate<Dtype>(&lrn_layout, dim, sizes, strides);
  }
  CHECK_EQ(e, E_SUCCESS);

  e = dnnLRNCreateForward<Dtype>(&lrnFwd, lrn_layout, size_, alpha_, beta_, k_);
  CHECK_EQ(e, E_SUCCESS);
  e = dnnLayoutCreateFromPrimitive<Dtype>(&lrn_buffer_l, lrnFwd, dnnResourceWorkspace);
  CHECK_EQ(e, E_SUCCESS);
  e = dnnAllocateBuffer<Dtype>((void **)&lrn_buffer_, lrn_buffer_l);
  CHECK_EQ(e, E_SUCCESS);
  e = dnnLRNCreateBackward<Dtype>(&lrnBwd, lrn_layout, lrn_layout, size_, alpha_, beta_, k_);
  CHECK_EQ(e, E_SUCCESS);

  dnnLayoutDelete<Dtype>(lrn_layout);
  dnnLayoutDelete<Dtype>(lrn_buffer_l);

}

template <typename Dtype>
void DnnLRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_ = bottom[0]->num();
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    top[0]->Reshape(num_, channels_, height_, width_);
    break;
  }
}

template <typename Dtype>
void DnnLRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_cpu(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void DnnLRNLayer<Dtype>::CrossChannelForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  void* bottom_data = (void*)bottom[0]->cpu_data();
  void* top_data = (void*)top[0]->mutable_cpu_data();

  dnnError_t e;
  void* lrn_res[dnnResourceNumber];
  lrn_res[dnnResourceSrc] = bottom_data;
  lrn_res[dnnResourceDst] = top_data;
  lrn_res[dnnResourceWorkspace] = lrn_buffer_;

  e = dnnExecute<Dtype>(lrnFwd, lrn_res);
  CHECK_EQ(e, E_SUCCESS);
}

template <typename Dtype>
void DnnLRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_cpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void DnnLRNLayer<Dtype>::CrossChannelBackward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  void* top_diff = (void*)top[0]->cpu_diff();
  void* bottom_data = (void*)bottom[0]->cpu_data();
  void* bottom_diff = (void*)bottom[0]->cpu_diff();

  dnnError_t e;
  void* lrn_res[dnnResourceNumber];
  lrn_res[dnnResourceSrc] = bottom_data;
  lrn_res[dnnResourceDiffDst] = top_diff;
  lrn_res[dnnResourceDiffSrc] = bottom_diff;
  lrn_res[dnnResourceWorkspace] = lrn_buffer_;

  e = dnnExecute<Dtype>(lrnBwd, lrn_res);
  CHECK_EQ(e, E_SUCCESS);
}


#ifdef CPU_ONLY
STUB_GPU(DnnLRNLayer);
STUB_GPU_FORWARD(DnnLRNLayer, CrossChannelForward);
STUB_GPU_BACKWARD(DnnLRNLayer, CrossChannelBackward);
#endif

INSTANTIATE_CLASS(DnnLRNLayer);
REGISTER_LAYER_CLASS(DnnLRN);

}  // namespace caffe
