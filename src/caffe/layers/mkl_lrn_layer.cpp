#ifdef MKL2017_SUPPORTED
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/mkl_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
MKLLRNLayer<Dtype>::~MKLLRNLayer() {
  dnnDelete<Dtype>(lrnFwd);
  dnnDelete<Dtype>(lrnBwd);
  dnnReleaseBuffer<Dtype>(lrn_buffer_);
  dnnLayoutDelete<Dtype>(layout_usr_);
}

template <typename Dtype>
void MKLLRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  e = dnnLayoutCreate<Dtype>(&layout_usr_, dim, sizes, strides);
  CHECK_EQ(e, E_SUCCESS);

  // Fwd, Bwd primitives and lrn_buffer_ are allocated in  "Lazy"
  // mode, because here we don't know
  // what layout is used by neighbours.
}

template <typename Dtype>
void MKLLRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void MKLLRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_cpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void MKLLRNLayer<Dtype>::CrossChannelForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const void* bottom_data =
    reinterpret_cast<const void*>(bottom[0]->prv_data());
  void* top_data = NULL;

  if (NULL != bottom_data) {
    // Is it the first pass? Create a primitive.
    if (lrnFwd == NULL) {
      CHECK_EQ((bottom[0]->get_prv_descriptor_data())->get_descr_type(),
              PrvMemDescr::PRV_DESCR_MKL2017);
      shared_ptr<MKLData<Dtype> > mem_descr
        =  boost::static_pointer_cast<MKLData<Dtype> >
              (bottom[0]->get_prv_descriptor_data());
      CHECK(mem_descr != NULL);

      dnnError_t e;
      dnnLayout_t lrn_buffer_l = NULL;

      e = dnnLRNCreateForward<Dtype>(
              &lrnFwd, NULL, mem_descr->layout_int, size_, alpha_, beta_, k_);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnLayoutCreateFromPrimitive<Dtype>(
              &lrn_buffer_l, lrnFwd, dnnResourceWorkspace);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnAllocateBuffer<Dtype>(
              reinterpret_cast<void **>(&lrn_buffer_), lrn_buffer_l);
      CHECK_EQ(e, E_SUCCESS);
      dnnLayoutDelete<Dtype>(lrn_buffer_l);

      fwd_top_data = mem_descr;
    }
    top_data = top[0]->mutable_prv_data();
    top[0]->set_prv_descriptor_data(fwd_top_data);

  } else {
    DLOG(INFO) << "Using cpu_data in MKLLRNLayer.";
    if (lrnFwd == NULL) {
      // First pass
      dnnError_t e;
      dnnLayout_t lrn_buffer_l = NULL;
      e = dnnLRNCreateForward<Dtype>(
              &lrnFwd, NULL, layout_usr_, size_, alpha_, beta_, k_);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnLayoutCreateFromPrimitive<Dtype>(
              &lrn_buffer_l, lrnFwd, dnnResourceWorkspace);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnAllocateBuffer<Dtype>(
              reinterpret_cast<void **>(&lrn_buffer_), lrn_buffer_l);
      CHECK_EQ(e, E_SUCCESS);
      dnnLayoutDelete<Dtype>(lrn_buffer_l);
    }
    bottom_data = reinterpret_cast<const void*>(bottom[0]->cpu_data());
    top_data = top[0]->mutable_cpu_data();
  }

  dnnError_t e;
  void* lrn_res[dnnResourceNumber];
  lrn_res[dnnResourceSrc] = const_cast<void*>(bottom_data);
  lrn_res[dnnResourceDst] = top_data;
  lrn_res[dnnResourceWorkspace] = lrn_buffer_;

  e = dnnExecute<Dtype>(lrnFwd, lrn_res);
  CHECK_EQ(e, E_SUCCESS);
}

template <typename Dtype>
void MKLLRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_cpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void MKLLRNLayer<Dtype>::CrossChannelBackward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const void* top_diff = reinterpret_cast<const void*>(top[0]->prv_diff());
  const void* bottom_data =
      reinterpret_cast<const void*>(bottom[0]->prv_data());
  void* bottom_diff = NULL;

  if (top_diff && bottom_data) {
    bottom_diff = reinterpret_cast<void*>(bottom[0]->mutable_prv_diff());
    // Is it the first pass? Create a primitive.
    if (lrnBwd == NULL) {
      CHECK_EQ((top[0]->get_prv_descriptor_diff())->get_descr_type(),
              PrvMemDescr::PRV_DESCR_MKL2017);
      shared_ptr<MKLDiff<Dtype> > mem_descr
        =  boost::static_pointer_cast<MKLDiff<Dtype> >
              (top[0]->get_prv_descriptor_diff());
      CHECK(mem_descr != NULL);

      dnnError_t e;
      e = dnnLRNCreateBackward<Dtype>(&lrnBwd, NULL, mem_descr->layout_int,
              mem_descr->layout_int, size_, alpha_, beta_, k_);
      CHECK_EQ(e, E_SUCCESS);

      bwd_bottom_diff = mem_descr;
    }
    bottom[0]->set_prv_descriptor_diff(bwd_bottom_diff);

  } else {
    DLOG(INFO) << "Using cpu_data in MKLLRNLayer.";
    top_diff = reinterpret_cast<const void*>(top[0]->cpu_diff());
    bottom_data = reinterpret_cast<const void*>(bottom[0]->cpu_data());
    bottom_diff = reinterpret_cast<void*>(bottom[0]->mutable_cpu_diff());
    if (lrnBwd == NULL) {
      dnnError_t e;
      e = dnnLRNCreateBackward<Dtype>(&lrnBwd, NULL, layout_usr_,
              layout_usr_, size_, alpha_, beta_, k_);
      CHECK_EQ(e, E_SUCCESS);
    }
  }

  dnnError_t e;
  void* lrn_res[dnnResourceNumber];
  lrn_res[dnnResourceSrc] = const_cast<void *>(bottom_data);
  lrn_res[dnnResourceDiffDst] = const_cast<void *>(top_diff);
  lrn_res[dnnResourceDiffSrc] = bottom_diff;
  lrn_res[dnnResourceWorkspace] = lrn_buffer_;

  e = dnnExecute<Dtype>(lrnBwd, lrn_res);
  CHECK_EQ(e, E_SUCCESS);
}


#ifdef CPU_ONLY
STUB_GPU(MKLLRNLayer);
STUB_GPU_FORWARD(MKLLRNLayer, CrossChannelForward);
STUB_GPU_BACKWARD(MKLLRNLayer, CrossChannelBackward);
#else
template <typename Dtype>
void MKLLRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLLRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLLRNLayer<Dtype>::CrossChannelForward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLLRNLayer<Dtype>::CrossChannelBackward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {NOT_IMPLEMENTED;}

#endif

INSTANTIATE_CLASS(MKLLRNLayer);
}  // namespace caffe
#endif  // #ifdef MKL2017_SUPPORTED
