#ifdef MKLDNN_SUPPORTED
#include <algorithm>
#include <vector>

#include "caffe/layers/mkldnn_layers.hpp"

namespace caffe {
template <typename Dtype>
MklDnnReLULayer<Dtype>::~MklDnnReLULayer() {
    dnnDelete<Dtype>(reluFwd_);
    dnnDelete<Dtype>(reluBwd_);
}

template <typename Dtype>
void MklDnnReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  CHECK_EQ(top[0]->shape(), bottom[0]->shape());
  size_t dim = bottom[0]->shape().size();
  size_t sizes[dim], strides[dim];
  for (size_t d = 0; d < dim; ++d) {
      sizes[d] = bottom[0]->shape()[d];
      strides[d] = (d == 0) ? 1 : strides[d-1]*sizes[d-1];
  }

  dnnError_t e;
  e = dnnLayoutCreate<Dtype>(&fwd_bottom_data_->layout_usr, dim, sizes,
          strides);
  CHECK_EQ(e, E_SUCCESS);
  e = dnnLayoutCreate<Dtype>(&bwd_top_diff_->layout_usr, dim, sizes, strides);
  CHECK_EQ(e, E_SUCCESS);

  // Names are for debugging only
  fwd_bottom_data_->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  bwd_top_diff_->name =    "bwd_top_diff      @ " + this->layer_param_.name();

  // "Lazy" allocation because here we don't know
  // what layout is used by neighbours.
  reluFwd_ = NULL;  // Will be allocated in a "lazy" way in first forward pass
  reluBwd_ = NULL;  // Will be allocated in a "lazy" way in first backward pass
}

template <typename Dtype>
void MklDnnReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  void* bottom_data =
    reinterpret_cast<void *>(const_cast<Dtype*>(bottom[0]->prv_data()));
  void* top_data = NULL;

  if (bottom_data) {
    top_data = top[0]->mutable_prv_data();

    if (reluFwd_ == NULL) {
      // first pass
      CHECK_EQ((bottom[0]->get_prv_descriptor_data())->get_descr_type(),
              PrvMemDescr::PRV_DESCR_MKLDNN);
      shared_ptr<MklDnnData<Dtype> > mem_descr
        =  boost::static_pointer_cast<MklDnnData<Dtype> >
              (bottom[0]->get_prv_descriptor_data());
      CHECK(mem_descr != NULL);

      Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
      dnnError_t e;
      e = dnnReLUCreateForward<Dtype>(&reluFwd_, NULL, mem_descr->layout_int,
              negative_slope);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnReLUCreateBackward<Dtype>(&reluBwd_, NULL, mem_descr->layout_int,
              mem_descr->layout_int, negative_slope);
      CHECK_EQ(e, E_SUCCESS);

      // copy shared_ptr
      fwd_bottom_data_ = mem_descr;

      e = dnnLayoutCreateFromPrimitive<Dtype>(&bwd_top_diff_->layout_int,
              reluFwd_, dnnResourceDst);
      CHECK_EQ(e, E_SUCCESS);
      bwd_top_diff_->create_conversions();
    }
  } else {
    DLOG(INFO) << "Using cpu_data in MklDnnReLULayer.";
    bottom_data =
      reinterpret_cast<void *>(const_cast<Dtype*>(bottom[0]->cpu_data()));
    top_data = top[0]->mutable_cpu_data();

    if (reluFwd_ == NULL) {
      // first pass
      dnnError_t e;
      Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
      e = dnnReLUCreateForward<Dtype>(&reluFwd_, NULL, fwd_bottom_data_->layout_usr,
              negative_slope);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnReLUCreateBackward<Dtype>(&reluBwd_, NULL, fwd_bottom_data_->layout_usr,
              fwd_bottom_data_->layout_usr, negative_slope);
      CHECK_EQ(e, E_SUCCESS);
    }
  }

  dnnError_t e;
  void* relu_res[dnnResourceNumber];
  relu_res[dnnResourceSrc] = bottom_data;
  relu_res[dnnResourceDst] = top_data;
  e = dnnExecute<Dtype>(reluFwd_, relu_res);
  CHECK_EQ(e, E_SUCCESS);
}

template <typename Dtype>
void MklDnnReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    void* top_diff =
      reinterpret_cast<void *>(const_cast<Dtype*>(top[0]->prv_diff()));
    void* bottom_data =
      reinterpret_cast<void *>(const_cast<Dtype*>(bottom[0]->prv_data()));
    void* bottom_diff = NULL;

    if (top_diff && bottom_data) {
      top_diff = bwd_top_diff_->get_converted_prv(top[0], true);
      bottom_diff = reinterpret_cast<void *>(bottom[0]->mutable_prv_diff());
    } else {
      DLOG(INFO) << "Using cpu_data in MklDnnReLULayer.";
      top_diff =
        reinterpret_cast<void *>(const_cast<Dtype*>(top[0]->cpu_diff()));
      bottom_data =
        reinterpret_cast<void *>(const_cast<Dtype*>(bottom[0]->cpu_data()));
      bottom_diff =
        reinterpret_cast<void *>(bottom[0]->mutable_cpu_diff());
    }

    dnnError_t e;
    void* relu_res[dnnResourceNumber];
    relu_res[dnnResourceSrc] = bottom_data;
    relu_res[dnnResourceDiffDst] = top_diff;
    relu_res[dnnResourceDiffSrc] = bottom_diff;
    e = dnnExecute<Dtype>(reluBwd_, relu_res);
    CHECK_EQ(e, E_SUCCESS);
  }
}

#ifdef CPU_ONLY
STUB_GPU(MklDnnReLULayer);
#else
template <typename Dtype>
void MklDnnReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MklDnnReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MklDnnReLULayer);
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
