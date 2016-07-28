#if defined(MKL2017_SUPPORTED)
#include <vector>

#include "caffe/layers/mkl_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
MKLSplitLayer<Dtype>::~MKLSplitLayer() {
  dnnDelete<Dtype>(sumPrimitive);
}

template <typename Dtype>
void MKLSplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_tops = top.size();
  size_t dim_src = bottom[0]->shape().size();

  size_t sizes_src[dim_src], strides_src[dim_src];
  for (size_t d = 0; d < dim_src; ++d) {
    sizes_src[d] = bottom[0]->shape()[dim_src - d - 1];
    strides_src[d] = (d == 0) ? 1 : strides_src[d-1]*sizes_src[d-1];
  }

  for (size_t i = 0; i < num_tops; ++i) {
    bwd_top_diff.push_back(shared_ptr<MKLDiff<Dtype> >(new MKLDiff<Dtype>));
    bwd_top_diff[i]->create_user_layout(dim_src, sizes_src, strides_src);
  }

  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(top.size(), 1);

  bwd_bottom_diff->create_user_layout(dim_src, sizes_src, strides_src);
}

template <typename Dtype>
void MKLSplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int count_ = bottom[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    // Do not allow in-place computation in the SplitLayer.  Instead, share data
    // by reference in the forward pass, and keep separate diff allocations in
    // the backward pass.  (Technically, it should be possible to share the diff
    // blob of the first split output with the input, but this seems to cause
    // some strange effects in practice...)
    CHECK_NE(top[i], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    top[i]->ReshapeLike(*bottom[0]);
    CHECK_EQ(count_, top[i]->count());
  }
}

template <typename Dtype>
void MKLSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void MKLSplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  dnnError_t e;
  vector<void*> top_diff;
  bool num_prv = 0;
  for (size_t i = 0; i < num_tops; i++) {
    top_diff.push_back(reinterpret_cast<void *>(
      const_cast<Dtype*>(top[i]->prv_diff())));
    if (top_diff[i] != NULL) {
      num_prv += 1;
    } else {
      top_diff[i] = reinterpret_cast<void*>(
      reinterpret_cast<void *>(const_cast<Dtype*>(top[i]->cpu_diff())));
    }
  }

  if (num_prv > 0) {
    if (sumPrimitive == NULL) {
      dnnLayout_t int_layout = NULL;
      for (size_t i = 0; i < num_tops; ++i) {
        if (top[i]->prv_diff() != NULL) {
          CHECK((top[i]->get_prv_descriptor_diff())->get_descr_type() ==
            PrvMemDescr::PRV_DESCR_MKL2017);
          shared_ptr<MKLDiff<Dtype> > mem_descr =
            boost::static_pointer_cast<MKLDiff<Dtype> >(
                top[i]->get_prv_descriptor_diff());
          CHECK(mem_descr != NULL);
          bwd_top_diff[i] = mem_descr;
          if (int_layout == NULL) {
            int_layout = mem_descr->layout_int;
          }
        }
      }
      e = dnnSumCreate<Dtype>(&sumPrimitive, NULL, num_tops,
        int_layout, &coeffs_[0]);
      CHECK_EQ(e, E_SUCCESS);

      bwd_bottom_diff->create_internal_layout(sumPrimitive, dnnResourceDst);

      for (size_t i = 0; i < num_tops; ++i) {
        if (top[i]->prv_diff() == NULL) {
          bwd_top_diff[i]->create_internal_layout(sumPrimitive,
                  (dnnResourceType_t)(dnnResourceMultipleSrc + i));
        }
      }
    }
  } else {
    if (sumPrimitive == NULL) {
      e = dnnSumCreate<Dtype>(&sumPrimitive, NULL, num_tops,
        bwd_bottom_diff->layout_usr, &coeffs_[0]);
      CHECK_EQ(e, E_SUCCESS);
    }
  }

  void *sum_res[dnnResourceNumber];
  for (int i = 0; i < num_tops; ++i) {
    if (bwd_top_diff[i]->convert_to_int) {
      sum_res[dnnResourceMultipleSrc + i] =
        bwd_top_diff[i]->get_converted_prv(top[i], false);
    } else {
      sum_res[dnnResourceMultipleSrc + i] =
        reinterpret_cast<void*>(top_diff[i]);
    }
  }

  if (bwd_bottom_diff->convert_from_int) {
    bottom[0]->set_prv_diff(bwd_bottom_diff->prv_ptr(),
        bwd_bottom_diff, false);
    sum_res[dnnResourceDst] =
        reinterpret_cast<void*>(bwd_bottom_diff->prv_ptr());
  } else {
    sum_res[dnnResourceDst] =
        reinterpret_cast<void*>(bottom[0]->mutable_cpu_diff());
  }

  e = dnnExecute<Dtype>(sumPrimitive, sum_res);
  CHECK_EQ(e, E_SUCCESS);
}

#ifdef CPU_ONLY
STUB_GPU(MKLSplitLayer);
#else
template <typename Dtype>
void MKLSplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLSplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLSplitLayer);
}  // namespace caffe
#endif  // #if defined(MKL2017_SUPPORTED)
