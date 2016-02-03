#include <algorithm>
#include <vector>

#include "caffe/layers/mkldnn_layers.hpp"
#include "dnn.h"

namespace caffe {
template <typename Dtype>
DnnReLULayer<Dtype>::~DnnReLULayer() {
    dnnDelete<Dtype>(&this->reluFwd);
    dnnDelete<Dtype>(&this->reluBwd);
}

template <typename Dtype>
void DnnReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

//  CHECK_EQ(top[0]->shape(), bottom[0]->shape());

  size_t dim = bottom[0]->shape().size();
  size_t sizes[dim], strides[dim];
  for (size_t d = 0; d < dim; ++d) {
      sizes[d] = bottom[0]->shape()[d];
      strides[d] = (d == 0) ? 1 : strides[d-1]*sizes[d-1];
  }

  dnnError_t e;
  dnnLayout_t relu_layout = NULL;
  e = dnnLayoutCreate<Dtype>(&relu_layout, dim, sizes, strides);
  CHECK_EQ(e, E_SUCCESS);

  dnnPrimitive_t reluFwd, reluBwd;
  e = dnnReLUCreateForward<Dtype>(&reluFwd, relu_layout, negative_slope);
  CHECK_EQ(e, E_SUCCESS);
  e = dnnReLUCreateBackward<Dtype>(&reluBwd, relu_layout, relu_layout, negative_slope);
  CHECK_EQ(e, E_SUCCESS);

  dnnLayoutDelete<Dtype>(relu_layout);

  this->reluFwd = reluFwd;
  this->reluBwd = reluBwd;
}

template <typename Dtype>
void DnnReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  void* bottom_data = (void*)bottom[0]->prv_data();
  void* top_data = NULL;

  if(NULL == bottom_data)
  {
    LOG(INFO) << "Using cpu_data in DnnReLULayer.";
    bottom_data = (void*)bottom[0]->cpu_data();
    top_data = top[0]->mutable_cpu_data();
  }
  else
  {
    //TODO: Add top converter
    //LOG(INFO) << "Using prv_data in DnnReLULayer.";
    top_data = top[0]->mutable_prv_data();
  }

  dnnError_t e;
  void* relu_res[dnnResourceNumber];
  relu_res[dnnResourceSrc] = bottom_data;
  relu_res[dnnResourceDst] = top_data;
  e = dnnExecute<Dtype>(this->reluFwd, relu_res);
  CHECK_EQ(e, E_SUCCESS);
}

template <typename Dtype>
void DnnReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    void* top_diff = (void*)top[0]->prv_diff();
    void* bottom_data = NULL;
    void* bottom_diff = NULL;

    if (NULL != top_diff) {
      bottom_data = (void*)bottom[0]->prv_data();
      bottom_diff = (void*)bottom[0]->mutable_prv_diff();

      if (NULL == bottom_data)
        LOG(FATAL) << "bottom_data is NULL";
    }else {
      top_diff = (void*)top[0]->cpu_diff();
      bottom_data = (void*)bottom[0]->cpu_data();
      bottom_diff = (void*)bottom[0]->mutable_cpu_diff();
    }

    dnnError_t e;
    void* relu_res[dnnResourceNumber];
    relu_res[dnnResourceSrc] = bottom_data;
    relu_res[dnnResourceDiffDst] = top_diff;
    relu_res[dnnResourceDiffSrc] = bottom_diff;
    e = dnnExecute<Dtype>(this->reluBwd, relu_res);
    CHECK_EQ(e, E_SUCCESS);
  }
}

#ifdef CPU_ONLY
STUB_GPU(DnnReLULayer);
#endif

INSTANTIATE_CLASS(DnnReLULayer);
REGISTER_LAYER_CLASS(DnnReLU);
}  // namespace caffe
