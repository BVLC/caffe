#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mkldnn_layers.hpp"

namespace caffe {


template <typename Dtype>
MklDnnLRNLayer<Dtype>::~MklDnnLRNLayer()
{
  dnnDelete<Dtype>(lrnFwd);
  dnnDelete<Dtype>(lrnBwd);
  dnnReleaseBuffer<Dtype>(lrn_buffer_);
}


template <typename Dtype>
void MklDnnLRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  size_ = this->layer_param_.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";

  alpha_ = this->layer_param_.lrn_param().alpha();
  beta_ = this->layer_param_.lrn_param().beta();
  k_ = this->layer_param_.lrn_param().k();

  lrn_buffer_ = NULL;

  // "Lazy" allocation because here we don't know
  // what layout is used by neighbours.
  lrnFwd = NULL; // Will be allocated in a "lazy" way in first forward pass
  lrnBwd = NULL; // Will be allocated in a "lazy" way in first backward pass
}

template <typename Dtype>
void MklDnnLRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
void MklDnnLRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void MklDnnLRNLayer<Dtype>::CrossChannelForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  void* bottom_data = (void*)bottom[0]->prv_data();
  void* top_data = NULL;

  if(NULL != bottom_data)
  {
    // Is it the first pass? Create a primitive.
    if (lrnFwd == NULL) {
      shared_ptr<MklDnnData<Dtype> > mem_descr
        =  boost::static_pointer_cast<MklDnnData<Dtype> > (bottom[0]->get_prv_descriptor_data());
      CHECK(mem_descr != NULL);

      dnnError_t e;
      dnnLayout_t lrn_buffer_l = NULL;

      e = dnnLRNCreateForward<Dtype>(&lrnFwd, mem_descr->layout_int, size_, alpha_, beta_, k_);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnLayoutCreateFromPrimitive<Dtype>(&lrn_buffer_l, lrnFwd, dnnResourceWorkspace);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnAllocateBuffer<Dtype>((void **)&lrn_buffer_, lrn_buffer_l);
      CHECK_EQ(e, E_SUCCESS);
      dnnLayoutDelete<Dtype>(lrn_buffer_l);

      fwd_top_data = mem_descr;
    }
    top_data = top[0]->mutable_prv_data();
    top[0]->set_prv_descriptor_data(fwd_top_data);

  } else {
    LOG(FATAL) << "Not yet implemented for default caffe data layout";
    LOG(INFO) << "Using cpu_data in MklDnnLRNLayer.";
    bottom_data = (void*)bottom[0]->cpu_data();
    top_data = top[0]->mutable_cpu_data();

  }


  dnnError_t e;
  void* lrn_res[dnnResourceNumber];
  lrn_res[dnnResourceSrc] = bottom_data;
  lrn_res[dnnResourceDst] = top_data;
  lrn_res[dnnResourceWorkspace] = lrn_buffer_;

  e = dnnExecute<Dtype>(lrnFwd, lrn_res);
  CHECK_EQ(e, E_SUCCESS);
}

template <typename Dtype>
void MklDnnLRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
void MklDnnLRNLayer<Dtype>::CrossChannelBackward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  void* top_diff = (void*)top[0]->prv_diff();
  void* bottom_data = NULL;
  void* bottom_diff = NULL;

  if (NULL != top_diff) {
    bottom_data = (void*)bottom[0]->prv_data();
    bottom_diff = (void*)bottom[0]->mutable_prv_diff();

    if (NULL == bottom_data)
      LOG(FATAL) << "bottom_data is NULL";

    // Is it the first pass? Create a primitive.
    if (lrnBwd == NULL) {
      shared_ptr<MklDnnDiff<Dtype> > mem_descr
        =  boost::static_pointer_cast<MklDnnDiff<Dtype> > (top[0]->get_prv_descriptor_diff());
      CHECK(mem_descr != NULL);

      dnnError_t e;
      e = dnnLRNCreateBackward<Dtype>(&lrnBwd, mem_descr->layout_int, mem_descr->layout_int, size_, alpha_, beta_, k_);
      CHECK_EQ(e, E_SUCCESS);

      bwd_bottom_diff = mem_descr;
    }
    bottom[0]->set_prv_descriptor_diff(bwd_bottom_diff);

  } else {
    LOG(FATAL) << "No implemented for default caffe data layout";
    top_diff = (void*)top[0]->cpu_diff();
    bottom_data = (void*)bottom[0]->cpu_data();
    bottom_diff = (void*)bottom[0]->mutable_cpu_diff();
  }

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
STUB_GPU(MklDnnLRNLayer);
STUB_GPU_FORWARD(MklDnnLRNLayer, CrossChannelForward);
STUB_GPU_BACKWARD(MklDnnLRNLayer, CrossChannelBackward);
#endif

INSTANTIATE_CLASS(MklDnnLRNLayer);
}  // namespace caffe
